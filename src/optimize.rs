fn sigma_evaluation(handtracking: Handtracking, params: &[f32]) -> f32 {

}

fn optimize_sigma(path: &str) {

    all_maxima = process_directory_for_maxima(path, sigma);

    do_gradient_descent(|params| signma_evaluation(handtracking, params), initial_params);
}

// f(x)

