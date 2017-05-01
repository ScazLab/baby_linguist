extern crate image;
extern crate time;
extern crate rustfft;
extern crate num;
extern crate glob;
#[macro_use]
extern crate conrod;

mod support;
mod tracking;
mod gui;
mod optimize;

use std::path::Path;
use std::f64;
use glob::glob;

use image::GenericImage;
use image::DynamicImage;

pub const BEST_SIGMA: f64 = 3.8;
pub const BEST_COEFFICIENTS: [f64; 9] = [200.0, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02];


fn rgb_to_yuv(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let (rf, gf, bf) = (r as f32, g as f32, b as f32);
    let y = 0.299 * rf + 0.587 * gf + 0.114 * bf;
    let u = -0.147 * rf - 0.289 * gf + 0.436 * bf + 128.0;
    let v = 0.615 * rf - 0.515 * gf - 0.100 * bf + 128.0;
    return (y as u8, u as u8, v as u8);
}

// From Skin Segmentation Using YUV and RGB Color Spaces
// Zaher Hamid Al-Tairi, Rahmita Wirza Rahmat, M. Iqbal Saripan, and
// Puteri Suhaiza Sulaiman
// J Inf Process Syst, Vol.10, No.2, pp.283~299, June 2014
// http://dx.doi.org/10.3745/JIPS.02.0002
fn is_skin(r: u8, g: u8, b: u8) -> bool {
    let (_, u, v) = rgb_to_yuv(r, g, b);

    return u > 80 && u < 130 && v > 136 && v < 200 && v > u && r > 80 && g > 30 && b > 15 &&
           (r as i16 - g as i16).abs() > 15;
}

// Approximation for a gaussian blur based on http://blog.ivank.net/fastest-gaussian-blur.html
// Original code is MIT licenced
// The idea is to use a series of moving average filters.
// Because the coefficient is identical for each element, you can use an accumulator
// and thus the overall time complexity is just O(n) in the number of pixels.
// Please note that we do modify the input_img buffer!
fn gaussian_box_blur(mut sigma: f64,
                     input_img: &mut Vec<f64>,
                     width: u32,
                     output_img: &mut Vec<f64>) {
    // prepare the output buffer
    let len = input_img.len();
    output_img.reserve(len);
    unsafe {
        output_img.set_len(len);
    }

    sigma /= 1.5; // arbitrary, since this approximation seems to have a larger effective sigma

    let sizes = box_blur_sizes(sigma);
    let height = len / width as usize;
    box_blur_pass(input_img, output_img, width as usize, height, sizes[0]);
    box_blur_pass(output_img, input_img, width as usize, height, sizes[1]);
    box_blur_pass(input_img, output_img, width as usize, height, sizes[2]);
}

fn diff_of_gaussians(sigma: f64,
                     k: f64,
                     input_img: &mut Vec<f64>,
                     width: u32,
                     output_img: &mut Vec<f64>) {
    let len = input_img.len();
    let mut input_buffer = input_img.clone();
    let mut smooth_buffer_2 = Vec::<f64>::with_capacity(len);
    unsafe {
        smooth_buffer_2.set_len(len);
    }

    gaussian_box_blur(sigma, &mut input_buffer, width, &mut smooth_buffer_2);
    gaussian_box_blur(sigma * k, &mut input_buffer, width, output_img);

    for i in 0..output_img.len() {
        output_img[i] = (smooth_buffer_2[i] - output_img[i]).max(0.);
    }
}

fn box_blur_pass(v_in: &mut Vec<f64>,
                 v_out: &mut Vec<f64>,
                 width: usize,
                 height: usize,
                 radius: usize) {
    v_out.copy_from_slice(&v_in[..]);
    box_blur_x(v_out, v_in, width, height, radius);
    box_blur_y(v_in, v_out, width, height, radius);
}

fn box_blur_x(v_in: &mut Vec<f64>,
              v_out: &mut Vec<f64>,
              width: usize,
              height: usize,
              radius: usize) {
    assert!(width >= radius);

    let factor = 1.0 / (radius + radius + 1) as f64;

    let mut index: usize = 0;
    for _ in 0..height {
        let first_val = v_in[index];
        let last_val = v_in[index + width - 1];
        // we count values beyond the edges as equal to the edge value
        let mut val = first_val * (radius + 1) as f64;
        // Keep track of the left and right sides of the moving average
        let mut left_i = index;
        let mut right_i = index + radius;
        // also add in the values including and to the right of the current element
        for i in 0..radius {
            val += v_in[index + i];
        }
        // Add in further elements to the right as we subtract the edge values
        for _ in 0..radius + 1 {
            val += v_in[right_i] - first_val;
            right_i += 1;
            v_out[index] = val * factor;
            index += 1;
        }
        // And so on, we no longer have any more edge values
        for _ in radius + 1..width - radius {
            val += v_in[right_i] - v_in[left_i];
            right_i += 1;
            left_i += 1;
            v_out[index] = val * factor;
            index += 1;
        }
        // And now add in the right edge values
        for _ in width - radius..width {
            val += last_val - v_in[left_i];
            left_i += 1;
            v_out[index] = val * factor;
            index += 1;
        }
    }
}

fn box_blur_y(v_in: &mut Vec<f64>,
              v_out: &mut Vec<f64>,
              width: usize,
              height: usize,
              radius: usize) {
    assert!(width >= radius);

    let factor = 1.0 / (radius + radius + 1) as f64;

    for x in 0..width {
        let mut index = x;
        let first_val = v_in[index];
        let last_val = v_in[index + (height - 1) * width];
        // we count values beyond the edges as equal to the edge value
        let mut val = first_val * (radius + 1) as f64;
        // Keep track of the "left"/top and "right"/bottom sides of the moving average
        let mut left_i = index;
        let mut right_i = index + radius * width;
        // also add in the values including and to the "right" of the current element
        for i in 0..radius {
            val += v_in[index + i * width];
        }
        // Add in further elements to the "right" as we subtract the edge values
        for _ in 0..radius + 1 {
            val += v_in[right_i] - first_val;
            right_i += width;
            v_out[index] = val * factor;
            index += width;
        }
        // And so on, we no longer have any more edge values
        for _ in radius + 1..height - radius {
            val += v_in[right_i] - v_in[left_i];
            right_i += width;
            left_i += width;
            v_out[index] = val * factor;
            index += width;
        }
        // And now add in the "right" edge values
        for _ in height - radius..height {
            val += last_val - v_in[left_i];
            left_i += width;
            v_out[index] = val * factor;
            index += width;
        }
    }
}

fn box_blur_sizes(sigma: f64) -> [usize; 3] {
    let mut sizes: [usize; 3] = [0; 3];
    let num_passes = sizes.len();

    let filter_width_ideal = f64::sqrt((12.0 * sigma * sigma / num_passes as f64) + 1.0);
    let mut width_low = filter_width_ideal as usize;
    if width_low % 2 == 0 {
        width_low -= 1; // must be odd
    }
    let width_high = width_low + 2;

    let mean_ideal = (12.0 * sigma * sigma - (num_passes * width_low * width_low) as f64 -
                      4.0 * (num_passes * width_low) as f64 -
                      3.0 * num_passes as f64) /
                     (-4.0 * width_low as f64 - 4.0);
    let mean = f64::round(mean_ideal) as usize;
    for i in 0..num_passes {
        sizes[i] = if i < mean { width_low } else { width_high };
    }
    return sizes;
}

#[allow(dead_code)]
fn benchmark<F: FnMut()>(name: &str, mut f: F) -> f64 {
    let start = time::precise_time_s();
    let mut repeat_number = 1;
    let mut total_runs = 0;
    while time::precise_time_s() - start < 2.0 {
        for _ in 0..repeat_number {
            f();
            total_runs += 1;
        }
        repeat_number *= 2;
    }
    let total_time = time::precise_time_s() - start;
    let time_each = total_time / total_runs as f64;
    println!("{} took: {:.4} seconds per call", name, time_each);
    return time_each;
}

fn find_local_maxima(in_img: &[f64], img_width: u32) -> Vec<(u32, u32, f64)> {
    let img_height = in_img.len() as u32 / img_width;
    let mut local_maxima = Vec::<(u32, u32, f64)>::new();

    for y in 1..img_height - 1 {
        'inner: for x in 1..(img_width - 1) {
            let curr_index = x + y * img_width;
            let curr_pixel = in_img[curr_index as usize];
            // Heres a heuristic im pulling out of my a**.
            // we dont want to check that every pixel is a
            // local max
            if curr_pixel > 0.225 {
                for adj_y in -1..2 as i32 {
                    for adj_x in -1..2 as i32 {
                        let adj_pixel = in_img[((x as i32 + adj_x) + (y as i32 + adj_y) * img_width as i32) as
                        usize];
                        if curr_pixel < adj_pixel {
                            // not the local max so lets move on
                            continue 'inner;
                        }
                    }
                }
                //println!("local max: {}, adj_pixels {:?}\n", curr_pixel,adj_pixels);
                local_maxima.push((x, y, curr_pixel));
            }
        }
    }
    return local_maxima;

}

fn skin_threshold(input_img: DynamicImage, grey_buffer: &mut Vec<f64>) {
    let (width, height) = input_img.dimensions();
    let len = (width * height) as usize;
    grey_buffer.reserve(len);
    unsafe {
        grey_buffer.set_len(len);
    }

    if let DynamicImage::ImageRgb8(rgb_img) = input_img {
        let rgb_buffer = rgb_img.into_raw();
        let mut in_index = 0;
        for i in 0..len {
            let (r, g, b) =
                (rgb_buffer[in_index], rgb_buffer[in_index + 1], rgb_buffer[in_index + 2]);
            grey_buffer[i] = if is_skin(r, g, b) { 1.0 } else { 0.0 };
            in_index += 3;
        }
    } else {
        panic!("Error: Input image must be RGB8!");
    }
}

#[allow(dead_code)]
fn write_grey_image(filename: &str, grey_img: &[f64], img_width: u32) {
    let len = grey_img.len();
    let img_height = len as u32 / img_width;
    let mut u8_buffer = Vec::<u8>::with_capacity(len);
    unsafe {
        u8_buffer.set_len(len);
    }

    for i in 0..len {
        u8_buffer[i] = (grey_img[i] * 255.0) as u8;
    }

    image::save_buffer(&Path::new(filename),
                       &u8_buffer[..],
                       img_width,
                       img_height,
                       image::Gray(8))
            .unwrap();
}

// This is the midpoint circle alg:
// https://en.wikipedia.org/wiki/Midpoint_circle_algorithm

fn draw_circles(maxima: &[(u32, u32)],
                radius: u32,
                in_img: &mut Vec<f64>,
                img_width: u32,
                grey_shade: f64) {
    let height = in_img.len() / img_width as usize;
    let width = img_width as usize;

    for m in maxima {

        let mut x = radius as usize;
        let mut y = 0 as usize;

        let x0 = m.0 as usize;
        let y0 = m.1 as usize;
        let mut err = 0 as i32;

        // if let DynamicImage::ImageRgb8(rgb_img) = in_img {
        //     let mut rgb_buffer = rgb_img.into_raw();
        while x >= y {
            //println!("width: {},height: {}\n", width, height);
            //println!("x: {},y: {},err: {}\n", x, y, err);
            let i1 = x0 + y;
            let i2 = y0 + x;
            let i3 = if x0 > y { x0 - y } else { 0 };
            let i4 = if x0 > x { x0 - x } else { 0 };
            let i5 = y0 + y;
            let i6 = if y0 > x { y0 - x } else { 0 };
            let i7 = if y0 > y { y0 - y } else { 0 };
            let i8 = x0 + x;

            if i8 < width && i5 < height {
                in_img[i8 + i5 * width] = grey_shade;
            }
            if i1 < width && i2 < height {
                in_img[i1 + i2 * width] = grey_shade;
            }
            if i3 > 0 && i2 < height {
                in_img[i3 + i2 * width] = grey_shade;
            }

            if i4 > 0 && i7 > 0 {
                in_img[i4 + i7 * width] = grey_shade;
            }
            if i3 > 0 && i6 > 0 {
                in_img[i3 + i6 * width] = grey_shade;
            }

            if i4 > 0 && i5 < height {
                in_img[i4 + i5 * width] = grey_shade;
            }
            if i1 < width && i6 > 0 {
                in_img[i1 + i6 * width] = grey_shade;
            }
            if i8 < width && i7 > 0 {
                in_img[i8 + i7 * width] = grey_shade;
            }
            //rgb_buffer[x as usize] = 0;

            if err <= 0 {
                y += 1;
                err += (2 * y + 1) as i32;
            } else {
                x -= 1;
                err -= (2 * x + 1) as i32;
            }
        }
    }
}

// Takes a vector  corresponding to the (x,y) coords
// of the the centroid of a hand in some number of frames
// and does a fourier analysis.
fn freq_analyze(window: Vec<(u32, u32)>) -> u32 {
    let fft_len = window.len() - 1;
    let mut fft = rustfft::FFT::new(fft_len, false);
    let mut signal = vec![num::Complex{re: 0.0, im: 0.0}; fft_len];

    for i in 0..fft_len {
        let (x_1, y_1) = window[i];
        let (x_2, y_2) = window[i + 1];

        let dx = x_2 as i32 - x_1 as i32;
        let dy = y_2 as i32 - y_1 as i32;

        signal[i].re = (dx * dx + dy * dy) as f64;


    }
    let mut spectrum = signal.clone();
    fft.process(&signal, &mut spectrum);
    println!("Freq spectrum: {:?}\n", spectrum);

    let mut max_i = 0;
    let mut max_mag = 0.0;

    for i in 1..spectrum.len() {
        let re = spectrum[i as usize].re;
        let im = spectrum[i as usize].im;
        let curr_mag = re * re + im * im;

        if curr_mag > max_mag {
            max_mag = curr_mag;
            max_i = i;
        }
    }
    // Place holder value. We need to account for the fact
    // that the baby may not be moving its hand
    if max_mag < 1000.0 {
        max_i = 0
    }
    println!("Max frequency is {}!\n", max_i);
    return max_i as u32;
}

fn compare_hand_freqs(left_hand_window: Vec<(u32, u32)>, right_hand_window: Vec<(u32, u32)>) {
    let left_hand_freq = freq_analyze(left_hand_window);
    let right_hand_freq = freq_analyze(right_hand_window);

    if left_hand_freq == right_hand_freq {
        println!("Both hands have the same Freq!")
    }

    let spectrum = 0.0;
    println!("Freq spectrum: {:?}\n", spectrum);
}

// Returns tuple of image's width, height, and the maxima
pub fn process_directory_for_maxima(path: &str,
                                    sigma: f64)
                                    -> (usize, usize, Vec<Vec<(u32, u32, f64)>>) {
    let mut all_maxima = Vec::<Vec<(u32, u32, f64)>>::new();

    // These vectors will be continually resused
    let mut grey_buffer = Vec::<f64>::new();
    let mut smooth_buffer = Vec::<f64>::new();

    let mut width: usize = 0;
    let mut height: usize = 0;

    for img_path in glob(&format!("{}/*.jpg", path)).expect("Failed to read glob pattern") {
        let img_path = img_path.unwrap();
        //println!("Processing image: {:?}", &img_path);
        let input_img = image::open(&img_path).unwrap();
        let (img_width, img_height) = input_img.dimensions();

        if width == 0 && height == 0 {
            width = img_width as usize;
            height = img_height as usize;
        } else {
            assert!(width == img_width as usize);
            assert!(height == img_height as usize);
        }

        skin_threshold(input_img, &mut grey_buffer);
        diff_of_gaussians(sigma,
                          1.6,
                          &mut grey_buffer,
                          width as u32,
                          &mut smooth_buffer);

        // Find and label the top maxima in the diff o' g. image
        let maxima = find_local_maxima(&mut smooth_buffer, width as u32);
        all_maxima.push(maxima);
    }

    return (width, height, all_maxima);
}

fn process_directory(path: &str,
                     baby_gui_skin: &mut Option<gui::BabyGui>,
                     baby_gui_hands: &mut Option<gui::BabyGui>) {
    // Parameters
    let sigma = 4.0;

    let mut track_hands = tracking::HandTracking::new();

    let mut total_process_time = 0.0;

    // These vectors will be continually resused
    let mut grey_buffer = Vec::<f64>::new();
    let mut smooth_buffer = Vec::<f64>::new();

    let mut i: u32 = 0;
    for img_path in glob(&format!("{}/*.jpg", path)).expect("Failed to read glob pattern") {
        let img_path = img_path.unwrap();
        println!("Processing image: {:?}", &img_path);
        let input_img = image::open(&img_path).unwrap();
        let (width, height) = input_img.dimensions();

        let start = time::precise_time_s();

        skin_threshold(input_img, &mut grey_buffer);
        diff_of_gaussians(sigma, 1.6, &mut grey_buffer, width, &mut smooth_buffer);

        // Find and label the top maxima in the diff o' g. image
        let maxima = find_local_maxima(&mut smooth_buffer, width);
        track_hands.add_maxima(width, height, &maxima, &BEST_COEFFICIENTS);

        total_process_time += time::precise_time_s() - start;

        // Draw all found points
        let feature_radius = (1.414 * sigma) as u32;
        let all_hand_points: Vec<(u32, u32)> = maxima.into_iter().map(|(x, y, _)| (x, y)).collect();
        draw_circles(&all_hand_points,
                     feature_radius,
                     &mut smooth_buffer,
                     width,
                     0.4);

        if track_hands.left_hand_coords.len() > 0 {
            let hand_points = vec![track_hands.left_hand_coords
                                       .last()
                                       .unwrap()
                                       .clone(),
                                   track_hands.right_hand_coords
                                       .last()
                                       .unwrap()
                                       .clone()];
            draw_circles(&hand_points, feature_radius, &mut smooth_buffer, width, 1.0);
        }

        //draw_circles(&maxima[0..cmp::min(4, maxima.len())], feature_radius, &mut smooth_buffer, width);

        if let &mut Some(ref mut gui_skin) = baby_gui_skin {
            gui_skin.set_image(&grey_buffer, width);
            gui_skin.handle_events();
        }

        if let &mut Some(ref mut gui_hands) = baby_gui_hands {
            gui_hands.set_image(&smooth_buffer, width);
            gui_hands.handle_events();
        }
        //write_grey_image(&format!("./video/DoG{}.png", i), &smooth_buffer[..], width);

        i += 1;
    }

    println!("Done! Processing {} frames took: {:.4} seconds per frame ({:.2} FPS)",
             i,
             total_process_time / i as f64,
             i as f64 / total_process_time);
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 2 {
        println!("Usage: {} [image directory]", &args[0]);
        return;
    }

    println!("Best hand coefs: {:?}", optimize::optimize_sigma(&args[1]));

    /*let mut baby_gui_skin = gui::BabyGui::new();
    let mut baby_gui_hands = gui::BabyGui::new();
    process_directory(&args[1], &mut baby_gui_skin, &mut baby_gui_hands);

    if let (Some(mut gui_skin), Some(mut gui_hands)) = (baby_gui_skin, baby_gui_hands) {
        while gui_skin.handle_events() && gui_hands.handle_events() {}
    }*/
}
