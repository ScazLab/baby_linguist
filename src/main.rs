extern crate image;

//use std::fs::File;
use std::path::Path;

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

fn main() {
    //let (y, u, v) = rgb_to_yuv(50, 100, 200);
    //println!("Y: {} U: {} V: {}\n", y, u, v);
    let input_image = image::open(&Path::new("baby3.jpg")).unwrap();
    let (width, height) = input_image.dimensions();
    let mut output_buffer = Vec::<u8>::with_capacity((width * height * 3) as usize);

    for (_, _, pixel) in input_image.pixels() {
        let (r, g, b) = (pixel.data[0], pixel.data[1], pixel.data[2]);
        if is_skin(r, g, b) {
            output_buffer.push(r);
            output_buffer.push(g);
            output_buffer.push(b);
        } else {
            output_buffer.push(0);
            output_buffer.push(0);
            output_buffer.push(0);
        }
    }

    image::save_buffer(&Path::new("baby3_filtered.png"), &output_buffer[..], width, height, image::RGB(8)).unwrap();

    println!("Done!");
}
