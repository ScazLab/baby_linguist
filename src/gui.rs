use support;
use std;

use conrod::{self, color, widget, Sizeable, Positionable, Widget, Colorable};
use conrod::backend::glium::glium;
use conrod::backend::glium::glium::{DisplayBuild, Surface};
use conrod::widget::primitive::text;

const WIDTH: u32 = 650;
const HEIGHT: u32 = 490;

widget_ids!(struct Ids { master,main, text,display_image, left_txt, right_txt });

pub struct BabyGui{
    display: glium::backend::glutin_backend::GlutinFacade,
    ui: conrod::Ui,
    renderer: conrod::backend::glium::Renderer,
    ids: Ids,
    event_loop: support::EventLoop,

    image_map: conrod::image::Map<glium::texture::Texture2d>,
    display_id: Option<conrod::image::Id>,

    width: f64,
    height: f64,

    left_freq: String,
    right_freq: String,
}

impl BabyGui {
    pub fn new() -> Option<Self> {
        let display = glium::glutin::WindowBuilder::new()
            .with_vsync()
            .with_dimensions(WIDTH, HEIGHT)
            .with_title("Baby Linguist")
            .build_glium();

        match display {
            Ok(display) => {
                let mut ui = conrod::UiBuilder::new([WIDTH as f64, HEIGHT as f64]).build();
                let renderer = conrod::backend::glium::Renderer::new(&display).unwrap();
                let ids = Ids::new(ui.widget_id_generator());

                Some(BabyGui {
                    display: display,
                    ui: ui,
                    renderer: renderer,
                    ids: ids,
                    event_loop: support::EventLoop::new(),
                    image_map: conrod::image::Map::new(),
                    display_id: None,
                    width: 0.,
                    height: 0.,
                    left_freq: "Left Freq: ".to_string(),
                    right_freq: "Right Freq: ".to_string(),
                     })
            }
            Err(e) => {
                println!("Unable to make window: {:?}", e);
                None
            }
        }
    }

    pub fn set_image(&mut self, grey_img: &[f64], width: u32) {
        let height = grey_img.len() as u32 / width;

        let len = (4 * width * height) as usize;
        let mut img_buffer = Vec::<u8>::with_capacity(len);
        unsafe {
            img_buffer.set_len(len);
        }

        let mut out_i = 0;
        for i in 0..(width * height) as usize {
            let val = (grey_img[i] * 255.0) as u8;
            img_buffer[out_i] = val;
            img_buffer[out_i + 1] = val;
            img_buffer[out_i + 2] = val;
            img_buffer[out_i + 3] = val;
            out_i += 4;
        }

        let raw_image = glium::texture::RawImage2d::from_raw_rgba_reversed(img_buffer,
                                                                           (width, height));
        let texture = glium::texture::Texture2d::new(&self.display, raw_image).unwrap();

        if let Some(display_id) = self.display_id {
            self.image_map.replace(display_id, texture);
        } else {
            self.display_id = Some(self.image_map.insert(texture));
        }
        self.event_loop.needs_update();
        self.ui.needs_redraw();

        self.width = width as f64;
        self.height = height as f64;
    }

    pub fn set_text(&mut self, left_freq: &str, right_freq: &str ){

        self.left_freq = left_freq.to_string();
        self.right_freq  = right_freq.to_string();

        //self.event_loop.needs_update();
        self.ui.needs_redraw();

    }
    // Returns true as long as the program should not quit
    pub fn handle_events(&mut self) -> bool {
        let font_path = "/usr/share/fonts/ttf-merriweather-ib/Merriweather-Black.ttf";
        self.ui.fonts .insert_from_file(font_path).unwrap();

        for event in self.event_loop.next(&self.display) {
            // Use the `winit` backend feature to convert the winit event to a conrod one,
            // to handle resizing and redrawing events
            if let Some(event) = conrod::backend::winit::convert(event.clone(), &self.display) {
                self.ui.handle_event(event);
            }

            match event {
                // Break from the loop upon `Escape`.
                glium::glutin::Event::KeyboardInput(_, _, Some(glium::glutin::VirtualKeyCode::Escape)) |
                glium::glutin::Event::KeyboardInput(_, _, Some(glium::glutin::VirtualKeyCode::Q)) |
                glium::glutin::Event::Closed =>
                    std::process::exit(0),
                _ => {}
            }
        }

        // Use an extra block so the mutable reference to self.ui is released
        {
            let widgets_ui = &mut self.ui.set_widgets();

            widget::Canvas::new()
                .color(color::BLACK)
                .w_h(self.width,self.height)
                .set(self.ids.master, widgets_ui);

            // widget::Canvas::new()
            //     .top_left_of(self.ids.master)
            //     .w_h(100.,50.)
            //     .color(color::BLUE)
            //     .set(self.ids.text,widgets_ui);

            widget::text_edit::TextEdit::new(&format!("{}\n{}",
                                                      &self.left_freq,
                                                      &self.right_freq))
                .color(color::WHITE)
                .font_size(12)
                .bottom_left_of(self.ids.master)
                .set(self.ids.left_txt, widgets_ui);

            println!("MAX LEFT FREQ: {:?}",&self.left_freq);

            if let Some(display_id) = self.display_id {
                // Instantiate the `Image` at its full size in the middle of the window.
                widget::Image::new(display_id)
                    .w_h(self.width, self.height)
                    .top_left_of(self.ids.master)
                    .set(self.ids.display_image, widgets_ui);

            }

        }

        // Render the `Ui` and then display it on the screen.
        if let Some(primitives) = self.ui.draw_if_changed() {
            self.renderer.fill(&self.display, primitives, &self.image_map);
            let mut target = self.display.draw();
            target.clear_color(0.0, 0.0, 0.0, 1.0);
            self.renderer.draw(&self.display, &mut target, &self.image_map).unwrap();
            target.finish().unwrap();
        }

        return true;
    }
}
