extern crate image;

//use std::fs::File;
use std::path::Path;
use std::f32;

use image::GenericImage;

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
    let img_height = in_img.len() as u32 / img_width;
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
                    value += in_img[((y - kernel_y) * img_width as i32 + (x - kernel_x)) as usize] * kernel[kernel_index];
                    kernel_index += 1;
                }
            }
            out_img[img_index] = value;
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

fn main() {
    //let (y, u, v) = rgb_to_yuv(50, 100, 200);
    //println!("Y: {} U: {} V: {}\n", y, u, v);
    let input_image = image::open(&Path::new("babysleeves.jpg")).unwrap();
    let (width, height) = input_image.dimensions();
    let mut output_buffer = Vec::<u8>::with_capacity((width * height * 3) as usize);

    let mut grey_buffer = Vec::<f32>::with_capacity((width * height) as usize);

    for (_, _, pixel) in input_image.pixels() {
        let (r, g, b) = (pixel.data[0], pixel.data[1], pixel.data[2]);
        if is_skin(r, g, b) {
            output_buffer.push(r);
            output_buffer.push(g);
            output_buffer.push(b);

            grey_buffer.push(1.0);
        } else {
            output_buffer.push(0);
            output_buffer.push(0);
            output_buffer.push(0);

            grey_buffer.push(r as f32 / 255.0);
        }
    }

    let mut smooth_buffer = Vec::<f32>::with_capacity((width * height) as usize);
    unsafe {
        let cap = smooth_buffer.capacity();
        smooth_buffer.set_len(cap);
    }

    let (kernel, kernel_dim) = gaussian_kernel(2.0);

    /*for i in 0..kernel.len() as usize {
        print!("{}, ", kernel[i]);
    }*/

    convolve(&grey_buffer[..], width, &kernel[..], kernel_dim, &mut smooth_buffer);

    let mut smooth_u8_buffer = Vec::<u8>::with_capacity((width * height) as usize);
    let mut grey_u8_buffer = Vec::<u8>::with_capacity((width * height) as usize);

    for i in 0..(width * height) as usize {
        smooth_u8_buffer.push((smooth_buffer[i] * 255.0) as u8);
        grey_u8_buffer.push((grey_buffer[i] * 255.0) as u8);
    }

    image::save_buffer(&Path::new("babysleeves_greyskin.png"), &grey_u8_buffer[..], width, height, image::Gray(8)).unwrap();

    let res = image::save_buffer(&Path::new("babysleeves_smooth.png"), &smooth_u8_buffer[..], width, height, image::Gray(8));
    match res {
        Ok(v) => println!("Writing the file worked!"),
        Err(e) => println!("Error writing image: {:?}", e),
    }

    println!("Done!");
}
