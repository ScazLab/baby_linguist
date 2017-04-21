fn evaluate_tracker(handtracker: HandTracking) -> f64 {
    let num_prev_coords = handtracker.left_hand_coords.len() - 1;
    let mut sum_least_sqrs = 0;

    for i in 0..num_prev_coords {
        let l_prev = handtracker.left_hand_coords[i];
        let r_prev = handtracker.right_hand_coords[i];
        let l_next = handtracker.left_hand_coords[i + 1];
        let r_next = handtracker.right_hand_coords[i + 1];

        let d_l = (l_prev.0 - l_next.0, l_prev.1 - l_next.1);
        let d_r = (r_prev.0 - r_next.0, r_prev.1 - r_next.1);

        sum_least_sqrs += d_l.0 * d_l.0 + d_l.1 * d_l.1 + d_r.0 * d_r.0 + d_r.1 * d_r.1;
    }

    return sum_least_sqrs as f64;
}

fn sigma_evaluation(path: &str, params: &[f64]) -> f64 {
    let (width, height, all_maxima) = process_directory_for_maxima(path, params[0]);
    let mut track_hands = tracking::HandTracking::new();
    for maxima in all_maxima {
        track_hands.add_maxima(width, height, &maxima, BEST_COEFFICIENTS);
    }
    return evaluate_tracker(track_hands);
}

fn optimize_sigma(path: &str) {
    return gradient_optimize(|params| sigma_evaluation(path, params), &[BEST_SIGMA])[0];
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
                f_plus_grad += alpha*t*(-grad[i])*grad[i];
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
// f(x)
