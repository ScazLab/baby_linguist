use std::f64;

fn evaluate_tracker(handtracker: ::tracking::HandTracking) -> (f64,f64) {
    //println!("Len of left hand coords: {}, right: {}", handtracker.left_hand_coords.len(), handtracker.right_hand_coords.len());
    if handtracker.left_hand_coords.len() == 0 {
        return (f64::MAX,f64::MAX);
    }

    let num_prev_coords = handtracker.left_hand_coords.len() - 1;
    let mut sum_least_sqrs_l = 0;
    let mut sum_least_sqrs_r = 0;

    //println!(" num coords  {:?}",handtracker.left_hand_coords);
    for i in 0..num_prev_coords {
        let l_prev = handtracker.left_hand_coords[i];
        let r_prev = handtracker.right_hand_coords[i];
        let l_next = handtracker.left_hand_coords[i + 1];
        let r_next = handtracker.right_hand_coords[i + 1];

        let d_l = (l_prev.0 as i32 - l_next.0 as i32, l_prev.1 as i32 - l_next.1 as i32);
        let d_r = (r_prev.0 as i32 - r_next.0 as i32, r_prev.1 as i32 - r_next.1 as i32);

        sum_least_sqrs_l += d_l.0 * d_l.0 + d_l.1 * d_l.1;
        sum_least_sqrs_r += d_r.0 * d_r.0 + d_r.1 * d_r.1;
    }

    return (sum_least_sqrs_l as f64, sum_least_sqrs_r as f64);
}

fn sigma_evaluation(path: &str, params: &[f64]) -> (f64,f64) {
    if params[0] <= 0.0 {
        return (f64::MAX,f64::MAX);
    }

    let (width, height, all_maxima) = ::process_directory_for_maxima(path, params[0]);
    if params[0] >= width as f64 / 2.0 {
        return (f64::MAX,f64::MAX);
    }

    let mut track_hands = ::tracking::HandTracking::new();
    for maxima in all_maxima {
        track_hands.add_maxima(width as u32, height as u32, &maxima, &::BEST_COEFFICIENTS);
    }
    return evaluate_tracker(track_hands);
}

pub fn optimize_sigma(path: &str) -> f64 {
    //return gradient_optimize(&|params| sigma_evaluation(path, params), &[::BEST_SIGMA])[0];
    let mut best_score = sigma_evaluation(path, &[1.0]);
    let mut best_sigma = 1.0;
    for i in 1..30 {
        let new_sigma = 1.0 + i as f64 * 0.1;
        let new_score = sigma_evaluation(path, &[new_sigma]);
        println!("Sigma {} yields score {:?}", new_sigma, new_score);
        if new_score < best_score {
            best_score = new_score;
            best_sigma = new_sigma;
        }
    }
    return best_sigma;
}

fn tracking_coefficients_evaluation(width: u32, height: u32,
                                    all_maxima: &Vec<Vec<(u32, u32, f64)>>,
                                    params: &[f64]) -> (f64,f64) {
    let mut track_hands = ::tracking::HandTracking::new();
    let mut param_array: [f64; 6] = [0.0; 6];
    param_array.copy_from_slice(params);
    for maxima in all_maxima {
        track_hands.add_maxima(width as u32, height as u32, maxima, &param_array);
    }
    return evaluate_tracker(track_hands);
}

pub fn optimize_tracking_coefficients(path: &str) -> Vec<f64> {
    let (width, height, all_maxima) = ::process_directory_for_maxima(path, ::BEST_SIGMA);
    //    return gradient_optimize(&|params| tracking_coefficients_evaluation(width as u32, height as u32, &all_maxima, params), &::BEST_COEFFICIENTS);
    return grid_search(&|params| tracking_coefficients_evaluation(width as u32, height as u32, &all_maxima, params),
                    &::BEST_COEFFICIENTS[..], &::SEARCH_RANGE, 100, 4);
}


fn calc_gradient<F>(func: &F, coeffs: &[f64]) -> Vec<f64>
    where F: Fn(&[f64]) -> (f64,f64)
{
    // First calculate numerical gradient
    //let h = 1e-11;
    let h = 1e-2;
    let coeff_len = coeffs.len();

    let mut grad = vec![0.0; coeff_len];

    for i in 0..coeff_len {
        let mut x_prev = coeffs.to_vec();
        let mut x_post = coeffs.to_vec();

        x_prev[i] -= h;
        x_post[i] += h;

        let f_prev = func(&x_prev[..]);
        let f_prev = f_prev.0 + f_prev.1;
        let f_post = func(&x_post[..]);
        let f_post = f_post.0 + f_post.1;
        //println!("x_post: {:?}\n x_prev: {:?}", x_post, x_prev);
        //println!("f_post: {} f_prev: {}", f_post, f_prev);

        grad[i] = -(f_post - f_prev) / (2.0 * h);
    }
    return grad;
}

fn grid_search<F>(func: &F, init_coeffs: &[f64], input_range: &[f64],
                  num_steps: u32, num_iters: u32)-> Vec<f64>
    where F: Fn(&[f64]) -> (f64,f64)
{

    let coeff_len = init_coeffs.len();
    assert!(coeff_len == input_range.len());
    let mut return_coeffs = init_coeffs.to_vec();
    let mut best_score = func(&return_coeffs[..]).0 + func(&return_coeffs[..]).1;

    for k in 0..num_iters{
        println!("----------Grid search pass {}----------",k);
        for i in 0..coeff_len{
            let mut best_coeff = return_coeffs[i];
            let mut offset = 0.0;
            // What is the search range for this coeff?
            let step_size = input_range[i] / num_steps as f64;

            for j in -(num_steps as i32)..(num_steps as i32) + 1{
                if j == 0{
                    continue;
                }
                let mut x_new = return_coeffs.to_vec();

                x_new[i] += (j + 1) as f64 * step_size;

                let f =  func(&x_new[..]);
                let f_new = f.0 + f.1;


                if f_new < best_score{
                    println!("Old score {:.3} f_new {:.3}" ,best_score, f_new);
                    best_score = f_new;
                    best_coeff = x_new[i];

                }
            }
            println!("Coeff {:.3} original: {:.3} new: {:.3}",i,
                     init_coeffs[i],best_coeff);
            return_coeffs[i] = best_coeff;
        }
    }
    return return_coeffs;
}

fn gradient_optimize<F>(func: &F, init_coeffs: &[f64]) -> Vec<f64>
    where F: Fn(&[f64]) -> (f64,f64)
{
    println!("Hello from Gradient Optimize!");

    let alpha = 0.01;
    let beta = 0.05;
    let eps = 1e-2;
    let coeff_len = init_coeffs.len();
    let mut return_coeffs = init_coeffs.to_vec();

    const MAX_OUTER_ITERS: u32 = 1000;
    const MAX_INNER_ITERS: u32 = 32;

    // Exit this loop when our grad is sufficiently small
    for _ in 0..MAX_OUTER_ITERS {
        let mut t = 1.0;
        let grad = calc_gradient(func, &return_coeffs[..]);
        println!("Our gradient is: {:?}", &grad);

        // is the norm of the grad sufficiently small?
        let mut grad_norm = 0.0;
        for i in &grad {
            grad_norm += i * i;
        }

        if grad_norm <= eps {
            break;
        }

        // Perform backtracking line search to calculate t


        for _ in 0..MAX_INNER_ITERS {
            let mut x_new = return_coeffs.clone();
            let f_plus_grad = func(&return_coeffs[..]);
            let mut f_plus_grad = f_plus_grad.0 + f_plus_grad.1;
            for i in 0..coeff_len {
                x_new[i] += t * grad[i];
                f_plus_grad += alpha * t * (-grad[i]) * grad[i];
            }

            let f_next = func(&x_new[..]);
            let f_next = f_next.0 + f_next.1;
            println!("F next: {}",f_next);

            if f_next < f_plus_grad {
                break;
            } else {
                t *= beta;
                println!("New t: {}",t);
                println!("F plus grad: {}",f_plus_grad);
            }
        }

        println!("\n Current score: {:?} with coeffs:\n {:?}\n", func(&return_coeffs[..]), &return_coeffs);

        // update coeffs to more optimal value
        for i in 0..coeff_len {
            return_coeffs[i] += t * grad[i];
        }
        println!("prev coefficients: {:?}",init_coeffs);
        println!("New coefficients: {:?}",return_coeffs);
    }
    return return_coeffs;
}
// f(x)
