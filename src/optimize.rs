fn evaluate_tracker(handtracker: HandTracking) -> f64 {
    let num_prev_coords = handtracker.left_hand_coords.len() - 1;
    let mut sum_least_sqrs = 0;

    for i in 0..num_prev_coords {
        let l_prev = handtracker.left_hand_coords[i];
        let r_prev = handtracker.right_hand_coords[i];
        let l_next = handtracker.left_hand_coords[i+1];
        let r_next = handtracker.right_hand_coords[i+1];

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

fn gradient_optimize(func: F, init_coeffs: &[f64]) -> Vec<f64> {
    let mut v: Vec<f64> = Vec::new();
    v.extend_from_slice(init_coeffs);
    return v;
}

// f(x)

