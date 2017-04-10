
const PREDICTION_POINTS: usize = 20;

pub struct HandTracking {
    pub left_hand_coords: Vec<(u32, u32)>,
    pub right_hand_coords: Vec<(u32, u32)>,
}

// From https://en.wikipedia.org/wiki/Simple_linear_regression
// We use the equation for beta-hat.
// We assume a constant spacing in time.
fn linear_interp_next(vals: &[(u32, u32)]) -> (u32, u32) {
    let mut avg_x: i32 = 0;
    let mut avg_y: i32 = 0;
    let avg_t = (vals.len() / 2) as i32;
    for &(x, y) in vals {
        avg_x += x as i32;
        avg_y += y as i32;
    }
    avg_x /= vals.len() as i32;
    avg_y /= vals.len() as i32;

    let mut sum_numer_x: i32 = 0;
    let mut sum_numer_y: i32 = 0;
    let mut sum_denom_x: i32 = 0;
    let mut sum_denom_y: i32 = 0;

    for t in 0..vals.len() as i32 {
        let (x, y) = vals[t as usize];
        let diff_x = x as i32 - avg_x;
        sum_numer_x += diff_x * (t - avg_t);
        sum_denom_x += diff_x * diff_x;

        let diff_y = y as i32 - avg_y;
        sum_numer_y += diff_y * (t - avg_t);
        sum_denom_y += diff_y * diff_y;
    }

    let slope_x = sum_numer_x / sum_denom_x;
    let slope_y = sum_numer_y / sum_denom_y;
    let &(last_x, last_y) = vals.last().unwrap();
    return ((last_x as i32 + slope_x) as u32, (last_y as i32 + slope_y) as u32);
}

impl HandTracking {
    pub fn new() -> Self {
        HandTracking {
            left_hand_coords: Vec::<(u32, u32)>::new(),
            right_hand_coords: Vec::<(u32, u32)>::new(),
        }
    }

    pub fn add_maxima(&mut self, width: u32, height: u32, maxima: &Vec<(u32, u32, f32)>, coeffs: &[f32; 9] ) {
        let mut maxima_by_x = maxima.clone();
        maxima_by_x.sort_by(|&(a, _, _), &(b, _, _)| a.partial_cmp(&b).unwrap() );

        // Determine which maxima are most likely to be each hand
        // Look for y-coords to be close to each other
        // Look for high scores
        // Look for values closish to image center
        let mut best_left: (u32, u32) = (0, 0);
        let mut best_right: (u32, u32) = (0, 0);
        let mut best_score: f32 = 0.0;

        for left_i in 0..maxima_by_x.len() {
            for right_i in left_i + 1..maxima_by_x.len() {
                // X coord, Y coord, and score
                let (l_x, l_y, l_s) = maxima_by_x[left_i];
                let (r_x, r_y, r_s) = maxima_by_x[right_i];
                // Maximize the total score

                // score of maxima is good
                let base_score = coeffs[0] * (-l_s - r_s);
                // We want the two points at about equal height
                // We square many quantities to minimize the effect of small differences
                let height_diff_score = coeffs[1] * (l_y as f32 - r_y as f32).powi(2);
                // They should be separated in x by some amount
                let sep_x_score = coeffs[2] * ((l_x as f32 - r_x as f32).abs() - width as f32 * 0.3).powi(2);
                // The center of the two hands should be near the center of the image
                let center_x_score = coeffs[3] * ((l_x + r_x) as f32 / 2.0 - width as f32 * 0.5).powi(2);
                let center_y_score = coeffs[4] * ((l_y + r_y) as f32 / 2.0 - height as f32 * 0.5).powi(2);
                let total_frame_score = base_score + height_diff_score + sep_x_score + center_x_score + center_y_score;

                // only print out competitive-ish scores, to make debugging easier
                if total_frame_score < 150.0 {
                    println!("Frame score of hand pair at at (({}, {}), ({}, {})) is {} + {} + {} + {} + {} = {}", l_x, l_y, r_x, r_y, base_score, height_diff_score, sep_x_score, center_x_score, center_y_score, total_frame_score);
                }

                let mut tracking_score = 0.0;

                // Predict the location from the past points
                let n = self.left_hand_coords.len();
                if n > PREDICTION_POINTS {
                    let (pred_left_x, pred_left_y) = linear_interp_next(&self.left_hand_coords[n-PREDICTION_POINTS..n]);
                    let (pred_right_x, pred_right_y) = linear_interp_next(&self.right_hand_coords[n-PREDICTION_POINTS..n]);
                    let (pred_left_x, pred_left_y) = (pred_left_x as f32, pred_left_y as f32);
                    let (pred_right_x, pred_right_y) = (pred_right_x as f32, pred_right_y as f32);
                    //let pred_left_x = self.left_hand_coords[n - 1].0 as f32 * 2.0 - self.left_hand_coords[n - 2].0 as f32;
                    //let pred_left_y = self.left_hand_coords[n - 1].1 as f32 * 2.0 - self.left_hand_coords[n - 2].1 as f32;
                    //let pred_right_x = self.right_hand_coords[n - 1].0 as f32 * 2.0 - self.right_hand_coords[n - 2].0 as f32;
                    //let pred_right_y = self.right_hand_coords[n - 1].1 as f32 * 2.0 - self.right_hand_coords[n - 2].1 as f32;
                    let err_left_x = coeffs[5] * (pred_left_x - l_x as f32).powi(2);
                    let err_left_y = coeffs[6] * (pred_left_y - l_y as f32).powi(2);
                    let err_right_x = coeffs[7] * (pred_right_x - r_x as f32).powi(2);
                    let err_right_y = coeffs[8] * (pred_right_y - r_y as f32).powi(2);
                    tracking_score = err_left_x + err_left_y + err_right_x + err_right_y;

                    // only print out competitive-ish scores, to make debugging easier
                    if total_frame_score < 150.0 {
                        println!("\tTracking score is {} + {} + {} + {} = {}",
                                 err_left_x, err_left_y, err_right_x, err_right_y, tracking_score);
                    }
                }

                let total_score = total_frame_score + tracking_score;

                if best_score == 0.0 || total_score < best_score {
                    best_score = total_score;
                    best_left = (l_x, l_y);
                    best_right = (r_x, r_y);
                }
            }
        }

        println!("Choose hand pair at (({}, {}), ({}, {})) with score of {}",
                 best_left.0, best_left.1, best_right.0, best_right.1, best_score);

        self.left_hand_coords.push(best_left);
        self.right_hand_coords.push(best_right);
    }
}

fn evaluate_tracker(handtracker: Handtracking)->f64{
    let num_prev_coords = handtracker.left_hand_coords.len()-1;
    let mut sum_least_sqrs = 0;

    for i in 0..num_prev_coords{
        let l_prev = handtracker.left_hand_coords[i];
        let r_prev = handtracker.right_hand_coords[i];
        let l_next = handtracker.left_hand_coords[i+1];
        let r_next = handtracker.right_hand_coords[i+1];

        let d_l = (l_prev.0 - l_next.0, l_prev.1 - l_next.1);
        let d_r = (r_prev.0 - r_next.0, r_prev.1 - r_next.1);

        sum_least_sqrs+= d_l.1*d_l.1 + d_l.1*d_l.1 + d_r.1*d_r.1 + d_r.1*d_r.1;
    }

    return sum_least_sqrs as f64;
}

fn calc_gradient(func:F, coeffs: &[f64])->Vec<f64>
    where F: Fn(&[f64])-> f64{
    // First calculate numerical gradient
    let h  = 10.pow(-11) as f64;
    let coeff_len = coeffs.len();

    let mut grad = Vec::with_capacity(coeff_len);

    for i in 0..coeff_len{
        let x_prev = coeff.clone();
        let x_post = coeff.clone();

        x_prev[i] -= h;
        x_post[i] += h;

        let f_prev = func(&x_prev[..]);
        let f_post = func(&x_post[..]);


        grad[i] =  -(f_post - f_prev)/(2.0 * h);
    }
    return grad;
}

fn gradient_optimize(func: F, init_coeffs: &[f64])-> Vec<f64>
    where F: Fn(&[f64])-> f64{

    let alpha = 0.2;
    let beta = 0.5; 
    let mut t = 1.0;
    let eps = 10.pow(-2);

    let coeff_len = coeffs.len();
    let mut return_coeffs = init_coeffs.clone();

    // Exit this loop when our grad is sufficiently small
    while true{

        let mut grad = calc_gradient(fun, return_coeffs);

        // Perform backtracking line search to calculate t
        let mut x_new = return_coeffs.clone();
        let mut f_plus_grad = func(&return_coeffs[..]);
        let mut keep_looping = true;

        while true{

            for i in 0..coeff_len{
                x_new[i] += t*grad[i];
                f_plus_grad += alpha*t*grad[i]*grad[i];
            }

            let f_next  = func(&x_new[..]);

            if f_next < f_plus_grad{ break; }

            else { t *= beta; }
        }

        // update coeffs to more optimal value
        for i in 0..coeff_len{
            return_coeffs[i] += t*grad[i];
        }

        grad = calc_gradient(func,&return_coeffs[..]);

        // is the norm of the grad sufficiently small?
        let grad_norm = 0.0;
        for i in grad{grad_norm += i*i;}

        if grad_norm <= eps{return return_coeffs;}
    }
}

