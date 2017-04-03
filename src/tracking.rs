pub struct HandTracking {
    pub left_hand_coords: Vec<(u32, u32)>,
    pub right_hand_coords: Vec<(u32, u32)>,
}

impl HandTracking {
    pub fn new() -> Self {
        HandTracking {
            left_hand_coords: Vec::<(u32, u32)>::new(),
            right_hand_coords: Vec::<(u32, u32)>::new(),
        }
    }

    pub fn add_maxima(&mut self, width: u32, height: u32, maxima: &Vec<(u32, u32, f32)>) {
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
                let base_score = -l_s - r_s +
                // We want the two points at about equal height
                           (l_y as f32 - r_y as f32).abs() +
                // They should be separated in x by some amount
                           ((l_x as f32 - r_x as f32).abs() + width as f32 * 0.2).abs() +
                // The center of the two hands should be near the center of the image
                           ((l_x + r_x) as f32 / 2.0 - width as f32 * 0.5).abs() +
                           ((l_y + r_y) as f32 / 2.0 - height as f32 * 0.5).abs();

                let mut tracking_score = 0.0;

                // Predict the location from the past points
                if self.left_hand_coords.len() > 2 {
                    let n = self.left_hand_coords.len();
                    let pred_left_x = self.left_hand_coords[n - 1].0 as f32 * 2.0 - self.left_hand_coords[n - 2].0 as f32;
                    let pred_left_y = self.left_hand_coords[n - 1].1 as f32 * 2.0 - self.left_hand_coords[n - 2].1 as f32;
                    let pred_right_x = self.right_hand_coords[n - 1].0 as f32 * 2.0 - self.right_hand_coords[n - 2].0 as f32;
                    let pred_right_y = self.right_hand_coords[n - 1].1 as f32 * 2.0 - self.right_hand_coords[n - 2].1 as f32;
                    tracking_score = (pred_left_x - l_x as f32).abs() + (pred_left_y - l_y as f32).abs() +
                                     (pred_right_x - r_x as f32).abs() + (pred_right_y - r_y as f32).abs();
                }

                let total_score = base_score + tracking_score;

                if total_score < best_score {
                    best_score = total_score;
                    best_left = (l_x, l_y);
                    best_right = (r_x, r_y);
                }
            }
        }

        self.left_hand_coords.push(best_left);
        self.right_hand_coords.push(best_right);
    }
}