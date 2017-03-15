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

fn convolve(in_img: &[f32], img_width: u32, kernel: &[f32], kernel_width: u32, out_img: &mut Vec<f32>) {
    // Guarentee space in and prepare the output buffer
    let img_len = in_img.len();
    out_img.reserve(img_len);
    unsafe {
        out_img.set_len(img_len);
    }

    let img_height = img_len as u32 / img_width;
    let kernel_height = kernel.len() as u32 / kernel_width;

    assert!(kernel_width % 2 == 1);
    assert!(kernel_height % 2 == 1);

    let half_k_width = kernel_width / 2;
    let half_k_height = kernel_height / 2;

    // This index points to the current pixel we want to output
    // for simplicity, we completely skip border pixels where the kernel
    // would not be completely within the image.
    let mut img_index: usize = (half_k_height * img_width) as usize;
    for y in half_k_height as i32..(img_height - half_k_height) as i32 {
        img_index += half_k_width as usize; // skip left border
        for x in half_k_width as i32..(img_width - half_k_width) as i32 {
            let mut value: f32 = 0.0;
            let mut kernel_index: usize = 0;
            // We start with the top-left corner of the kernel
            // and thus with the bottom-right corner of the image
            // We start just at an extra +1 x and y since we reduce the index before using it.
            let mut in_index = img_index + ((half_k_height + 1) * img_width + half_k_width + 1) as usize;
            for _ in 0..kernel_height as i32 {
                in_index -= img_width as usize;
                for _ in 0..kernel_width as i32 {
                    in_index -= 1;
                    value += in_img[in_index] * kernel[kernel_index];
                    kernel_index += 1;
                }
                in_index += kernel_width as usize; // rewind the input image index back to the right edge
            }
            out_img[img_index] = value;
            img_index += 1;
        }

        img_index += half_k_width as usize; // skip right border
    }
}

fn gaussian_kernel(sigma: f32) -> (Vec<f32>, u32) {
    let half_dim = f32::round(3.0 * sigma) as i32;
    let dim = (half_dim * 2 + 1) as u32;

    let len = (dim * dim) as usize;
    let mut kernel = Vec::<f32>::with_capacity(len);
    unsafe {
        kernel.set_len(len);
    }

    let mut kernel_index = 0;
    let scalar1: f32 = 1.0 / (2.0 * f32::consts::PI * sigma * sigma);
    let scalar2: f32 = 1.0 / (2.0 * sigma * sigma);
    for y in -half_dim..(half_dim + 1) {
        for x in -half_dim..(half_dim + 1) {
            kernel[kernel_index] = scalar1 * f32::exp(-((x * x + y * y) as f32) * scalar2);
            kernel_index += 1;
        }
    }

    return (kernel, dim);
}

fn gaussian_kernel_2d(sigma: f32) -> (Vec<f32>, u32) {
    let half_dim = f32::round(3.0 * sigma) as i32;
    let dim = (half_dim * 2 + 1) as u32;

    let mut kernel = Vec::<f32>::with_capacity(dim as usize);
    unsafe {
        kernel.set_len(dim as usize);
    }

    let mut kernel_index = 0;
    let scalar1: f32 = 1.0 / (f32::sqrt(2.0 * f32::consts::PI) * sigma);
    let scalar2: f32 = 1.0 / (2.0 * sigma * sigma);
    for x in -half_dim..(half_dim + 1) {
        kernel[kernel_index] = scalar1 * f32::exp(-((x * x) as f32) * scalar2);
        kernel_index += 1;
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

fn bench_convolve() {
    let base_img = image::open(&Path::new("babysleeves.jpg")).unwrap();
    let (width, height) = base_img.dimensions();

    let mut in_img = Vec::<f32>::with_capacity((width * height) as usize);
    skin_threshold(base_img, &mut in_img);

    let mut out_img = Vec::<f32>::with_capacity((width * height) as usize);
    let (kernel, kernel_dim) = gaussian_kernel(2.0);

    let img_len = in_img.len();
    out_img.reserve(img_len);
    unsafe {
        out_img.set_len(img_len);
    }

    let img_width = width;
    let kernel_width = kernel_dim;

    let img_height = img_len as u32 / img_width;
    let kernel_height = kernel.len() as u32 / kernel_width;

    assert!(kernel_width % 2 == 1);
    assert!(kernel_height % 2 == 1);

    let half_k_width = kernel_width / 2;
    let half_k_height = kernel_height / 2;

    benchmark("Convolve 1", || {
        let mut img_index: usize = (half_k_height * img_width) as usize;
        for y in half_k_height as i32..(img_height - half_k_height) as i32 {
            img_index += half_k_width as usize;

            for x in half_k_width as i32..(img_width - half_k_width) as i32 {
                let mut value: f32 = 0.0;
                let mut kernel_index: usize = 0;
                for kernel_y in -(half_k_height as i32)..(half_k_height + 1) as i32 {
                    for kernel_x in -(half_k_width as i32)..(half_k_width + 1) as i32 {
                        value += in_img[((y - kernel_y) * img_width as i32 + (x - kernel_x)) as usize] * kernel[kernel_index];
                        kernel_index += 1;
                    }
                }
                out_img[img_index] = value;
                img_index += 1;
            }

            img_index += half_k_width as usize;
        }
    });

    benchmark("Convolve 2", || {
        // This index points to the current pixel we want to output
        // for simplicity, we completely skip border pixels where the kernel
        // would not be completely within the image.
        let mut img_index: usize = (half_k_height * img_width) as usize;
        for y in half_k_height as i32..(img_height - half_k_height) as i32 {
            img_index += half_k_width as usize; // skip left border
            for x in half_k_width as i32..(img_width - half_k_width) as i32 {
                let mut value: f32 = 0.0;
                let mut kernel_index: usize = 0;
                // We start with the top-left corner of the kernel
                // and thus with the bottom-right corner of the image
                // We start just at an extra +1 x and y since we reduce the index before using it.
                let mut in_index = img_index + ((half_k_height + 1) * img_width + half_k_width + 1) as usize;
                for _ in 0..kernel_height as i32 {
                    in_index -= img_width as usize;
                    for _ in 0..kernel_width as i32 {
                        in_index -= 1;
                        value += in_img[in_index] * kernel[kernel_index];
                        kernel_index += 1;
                    }
                    in_index += kernel_width as usize; // rewind the input image index back to the right edge
                }
                out_img[img_index] = value;
                img_index += 1;
            }

            img_index += half_k_width as usize; // skip right border
        }
    });
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
    //bench_convolve();
    //return;

    let input_img = image::open(&Path::new("babysleeves.jpg")).unwrap();
    let (width, height) = input_img.dimensions();

    let mut grey_buffer = Vec::<f32>::with_capacity((width * height) as usize);
    skin_threshold(input_img, &mut grey_buffer);

    let mut smooth_buffer_intermediate = Vec::<f32>::with_capacity((width * height) as usize);
    let mut smooth_buffer = Vec::<f32>::with_capacity((width * height) as usize);
    let (kernel_2d, kernel_dim) = gaussian_kernel_2d(20.0);
    // The gaussian filter is separable, so we compute it in separate x and y steps
    // X step
    convolve(&grey_buffer[..], width, &kernel_2d[..], kernel_dim, &mut smooth_buffer_intermediate);
    // Y step
    convolve(&smooth_buffer_intermediate[..], width, &kernel_2d[..], 1, &mut smooth_buffer);

    write_grey_image("babysleeves_greyskin.png", &grey_buffer[..], width);
    write_grey_image("babysleeves_smooth.png", &smooth_buffer[..], width);

    println!("Done!");
}
