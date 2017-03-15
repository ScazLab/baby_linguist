extern crate image;
extern crate time;

//use std::fs::File;
use std::path::Path;
use std::f32;
use std::io::Write;

use image::GenericImage;
use image::DynamicImage;

fn rgb_to_yuv(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let (rf, gf, bf) = (r as f32, g as f32, b as f32);
    let y = 0.299 * rf + 0.587 * gf + 0.114 * bf;
    let u = -0.147 * rf - 0.289 * gf + 0.436 * bf + 128.0;
    let v = 0.615 * rf - 0.515 * gf - 0.100 * bf + 128.0;
    return (y as u8, u as u8, v as u8);
}

fn is_skin(r: u8, g: u8, b: u8) -> bool {
    let (_, u, v) = rgb_to_yuv(r, g, b);

    return u > 80 && u < 130 &&
           v > 136 && v < 200 &&
           v > u &&
           r > 80 && g > 30 && b > 15 &&
          (r as i16 - g as i16).abs() > 15;
}

fn convolve(in_image: &[f32], img_width: u32, kernel: &[f32], kernel_width: u32, out_image: &mut Vec<f32>) {
    let img_height = in_image.len() as u32 / img_width;
    let kernel_height = kernel.len() as u32 / kernel_width;

    assert!(kernel_width % 2 == 1);
    assert!(kernel_height % 2 == 1);

    let half_k_width = kernel_width / 2;
    let half_k_height = kernel_height / 2;

    let mut img_index: usize = (half_k_height * img_width) as usize;
    for y in half_k_height as i32..(img_height - half_k_height) as i32 {
        img_index += half_k_width as usize;

        for x in half_k_width as i32..(img_width - half_k_width) as i32 {
            let mut value: f32 = 0.0;
            let mut kernel_index: usize = 0;
            for kernel_y in -(half_k_height as i32)..(half_k_height + 1) as i32 {
                for kernel_x in -(half_k_width as i32)..(half_k_width + 1) as i32 {
                    value += in_image[((y - kernel_y) * img_width as i32 + (x - kernel_x)) as usize] * kernel[kernel_index];
                    kernel_index += 1;
                }
            }
            out_image[img_index] = value;
            img_index += 1;
        }

        img_index += half_k_width as usize;
    }
}

fn gaussian_kernel(sigma: f32) -> (Vec<f32>, u32) {
    let half_dim = f32::round(3.0 * sigma) as i32;
    let dim = (half_dim * 2 + 1) as u32;

    let mut kernel = Vec::<f32>::with_capacity((dim * dim) as usize);
    let scalar1: f32 = 1.0 / (2.0 * f32::consts::PI * sigma * sigma);
    let scalar2: f32 = 1.0 / (2.0 * sigma * sigma);
    for y in -half_dim..(half_dim + 1) {
        for x in -half_dim..(half_dim + 1) {
            kernel.push(scalar1 * f32::exp(-((x * x + y * y) as f32) * scalar2));
        }
    }

    return (kernel, dim);
}

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

fn bench_skin_threshold() {
    let input_img = image::open(&Path::new("babysleeves.jpg")).unwrap();
    let (width, height) = input_img.dimensions();
    let mut grey_buffer = Vec::<f32>::with_capacity((width * height) as usize);

    benchmark("Skin thresholding 1", || {
        grey_buffer.clear();
        for (_, _, pixel) in input_img.pixels() {
            let (r, g, b) = (pixel.data[0], pixel.data[1], pixel.data[2]);
            if is_skin(r, g, b) {
                grey_buffer.push(1.0);
            } else {
                grey_buffer.push(0.0);
            }
        }
    });

    benchmark("Skin thresholding 2", || {
        let mut out_index = 0;
        for (_, _, pixel) in input_img.pixels() {
            let (r, g, b) = (pixel.data[0], pixel.data[1], pixel.data[2]);
            if is_skin(r, g, b) {
                grey_buffer[out_index] = 1.0;
            } else {
                grey_buffer[out_index] = 0.0;
            }
            out_index += 1;
        }
    });

    benchmark("Skin thresholding 3", || {
        let mut out_index = 0;
        for (_, _, pixel) in input_img.pixels() {
            let (r, g, b) = (pixel.data[0], pixel.data[1], pixel.data[2]);
            grey_buffer[out_index] = if is_skin(r, g, b) { 1.0 } else { 0.0 };
            out_index += 1;
        }
    });

    if let DynamicImage::ImageRgb8(rgb_img) = input_img {
        let rgb_buffer = rgb_img.into_raw();
        benchmark("Skin thresholding 4", || {
            let mut in_index = 0;
            for i in 0..(width * height) as usize {
                let (r, g, b) = (rgb_buffer[in_index], rgb_buffer[in_index + 1], rgb_buffer[in_index + 2]);
                grey_buffer[i] = if is_skin(r, g, b) { 1.0 } else { 0.0 };
                in_index += 3;
            }
        });
    } else {
        writeln!(&mut std::io::stderr(), "Error: The input image was not RGB8! Uh oh...").unwrap();
    }
}

fn skin_threshold(input_img: DynamicImage, grey_buffer: &mut Vec<f32>) {
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
            let (r, g, b) = (rgb_buffer[in_index], rgb_buffer[in_index + 1], rgb_buffer[in_index + 2]);
            grey_buffer[i] = if is_skin(r, g, b) { 1.0 } else { 0.0 };
            in_index += 3;
        }
    } else {
        panic!("Error: Input image must be RGB8!");
    }
}

fn write_grey_image(filename: &str, grey_img: &[f32], img_width: u32) {
    let len = grey_img.len();
    let img_height = len as u32 / img_width;
    let mut u8_buffer = Vec::<u8>::with_capacity(len);
    unsafe {
        u8_buffer.set_len(len);
    }

    for i in 0..len {
        u8_buffer[i] = (grey_img[i] * 255.0) as u8;
    }

    image::save_buffer(&Path::new(filename), &u8_buffer[..], img_width, img_height, image::Gray(8)).unwrap();
}

fn main() {
    //bench_skin_threshold();
    //return;

    //let (y, u, v) = rgb_to_yuv(50, 100, 200);
    //println!("Y: {} U: {} V: {}\n", y, u, v);
    let input_img = image::open(&Path::new("babysleeves.jpg")).unwrap();
    let (width, height) = input_img.dimensions();

    let mut grey_buffer = Vec::<f32>::with_capacity((width * height) as usize);
    skin_threshold(input_img, &mut grey_buffer);

    let mut smooth_buffer = Vec::<f32>::with_capacity((width * height) as usize);
    unsafe {
        let cap = smooth_buffer.capacity();
        smooth_buffer.set_len(cap);
    }

    let (kernel, kernel_dim) = gaussian_kernel(2.0);

    convolve(&grey_buffer[..], width, &kernel[..], kernel_dim, &mut smooth_buffer);

    write_grey_image("babysleeves_greyskin.png", &grey_buffer[..], width);
    write_grey_image("babysleeves_smooth.png", &smooth_buffer[..], width);

    println!("Done!");
}
