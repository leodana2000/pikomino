/*
On ne peut pas juste faire le calcul sur tous les tirages possibles. Pour faire la pondération, on va calculer :
1. La probabilité d'obtenir un lancer ne comportant aucun dé que l'on puisse prendre
2. La probabilité d'obtenir chaque valeur de dé pour chaque dé disponible. Ensuite, on pondère la valeur du choix de prendre
k dés de valeur n, par la probabilité d'obtenir 

*/
#[macro_use]
extern crate lazy_static;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::convert::TryInto;


// Type aliases for caches
type ComputeRewardCache = HashMap<([usize; 6], usize), (f64, usize)>;
type PositionValueCache = HashMap<([usize; 6], usize, usize), f64>;
type FactorialCache = HashMap<usize, usize>;

fn compute_reward_if_leave(
    current_score: &usize,
    picked_dice: &[usize; 6],
    rewards: &[f64; 41],
    leaving_penalty: &f64,
    cache: &mut ComputeRewardCache,) -> (f64, usize) {
    let cache_key = (*picked_dice, *current_score);
    if let Some(&cached_result) = cache.get(&cache_key) {
        return cached_result;
    }

    let mut reward: f64 = *leaving_penalty;
    let mut max_pickomino_index_so_far: usize = 0;
    if picked_dice[5] != 0 {
        for p in 0..(*current_score + 1) {
            if rewards[p] > reward {
                reward = rewards[p];
                max_pickomino_index_so_far = p;
            }
        }
    }

    cache.insert(cache_key, (reward, max_pickomino_index_so_far));
    (reward, max_pickomino_index_so_far)
}

fn position_value(
    current_score: &usize,
    picked_dice: &[usize; 6],
    remaining_dice: &usize,
    list_throws: &Vec<[usize; 6]>,
    list_intervals: &Vec<usize>,
    rewards: &[f64; 41],
    leaving_penalty: &f64,
    cache: &mut PositionValueCache,) -> f64 {
    let cache_key = (*picked_dice, *current_score, *remaining_dice);
    if let Some(&cached_value) = cache.get(&cache_key) {
        return cached_value;
    }

    let mut values_vector: Vec<Vec<f64>> = vec![vec![]; 6];

    for dice in 0..6 {
        if picked_dice[dice] != 1 {
            for how_many_of_this_dice in 1..(*remaining_dice + 1) {
                let new_score = *current_score + how_many_of_this_dice * std::cmp::min(5, dice + 1);
                let mut new_picked_dice = picked_dice.clone();
                new_picked_dice[dice] = 1;
                let new_remaining_dice = remaining_dice - how_many_of_this_dice;
                values_vector[dice].push(position_value(
                    &new_score,
                    &new_picked_dice,
                    &new_remaining_dice,
                    list_throws,
                    list_intervals,
                    rewards,
                    leaving_penalty,
                    cache,
                ));
            }
        }
    }

    let interval_start = 8 - *remaining_dice;
    let mut total_reward = 0.0;
    for throw in list_intervals[interval_start]..list_intervals[interval_start + 1] {
        let throw_values = list_throws[throw];
        let mut max_reward_so_far = *leaving_penalty;
        for dice in 0..6 {
            if throw_values[dice] != 0 && picked_dice[dice] != 1 {
                let reward = values_vector[dice][throw_values[dice] - 1];
                if reward > max_reward_so_far {
                    max_reward_so_far = reward;
                }
            }
        }
        total_reward += max_reward_so_far*&LIST_PROBABILITIES[throw];
    }
    let average_reward = total_reward;

    let mut compute_reward_cache = HashMap::new();
    let reward_leave = compute_reward_if_leave(current_score, picked_dice, rewards, leaving_penalty, &mut compute_reward_cache).0;

    let result = average_reward.max(reward_leave);
    cache.insert(cache_key, result);
    result
}

#[pyfunction]
fn position_value_py(
    current_score: usize,
    picked_dice: Vec<usize>,
    remaining_dice: usize,
    rewards: Vec<f64>,
    leaving_penalty: f64
) -> PyResult<f64> {
    let mut picked_dice_arr: [usize; 6] = [0; 6];
    picked_dice_arr.copy_from_slice(&picked_dice[..6]);

    let mut rewards_arr: [f64; 41] = [0.0; 41];
    rewards_arr.copy_from_slice(&rewards[..41]);

    // Initialize the cache
    let mut cache = HashMap::new();

    // Call the Rust position_value function
    Ok(position_value(
        &current_score,
        &picked_dice_arr,
        &remaining_dice,
        &LIST_THROWS,      // Assuming LIST_THROWS and LIST_INTERVALS are defined with lazy_static!
        &LIST_INTERVALS,
        &rewards_arr,
        &leaving_penalty,
        &mut cache
    ))
}

fn factorial(n:usize, factorial_cache: &mut FactorialCache) -> usize {
    // Use the memoization with factorial_cache
    // If n is in the cache, return the value
    if n == 0 {
        return 1;
    }
    if let Some(&cached_result) = factorial_cache.get(&n) {
        return cached_result; 
        } else {
            let mut result: usize = n*factorial(n-1, factorial_cache);
            factorial_cache.insert(n, result);
            return result;
    }
}

fn multinomial_coefficient(throw_tuple:&[usize;6]) -> usize {
    let mut factorial_cache = FactorialCache::new();
    let mut result = factorial(throw_tuple.iter().sum(), &mut factorial_cache);
    for i in 0..6 {
        result /= factorial(throw_tuple[i], &mut factorial_cache);
    }
    result
}

fn compute_throws_probabilities(list_throws:&Vec<[usize;6]>, list_intervals:&Vec<usize>,) -> Vec<f64> {
    /*
    Computes the probability of obtaining any tuple.
    If we throw k dice, the probability of obtaining a tuple (a_1, ..., a_k) is given by:
    (k choose a_1, ..., a_k) * (1/6)^k
    */
    let mut vec_coefficients = Vec::new();
    for i in 0..list_intervals.len()-1 {
        let division_constant = (6 as f64).powi(8-i as i32);
        for j in list_intervals[i]..list_intervals[i+1] {
            let result = (multinomial_coefficient(&list_throws[j]) as f64)/division_constant;
            vec_coefficients.push(result);
        }
        // Debug check that the coefficients over the interval list_intervals[i]..list_intervals[i+1] sum to 1
        let sum: f64 = vec_coefficients.iter().sum();
        //println!("sum {}", sum);
    }
    vec_coefficients
}


fn compute_best_path(current_score:&usize, picked_dice:&[usize;6], remaining_dice:&usize, throw_result:&usize, list_throws:&Vec<[usize;6]>, list_intervals:&Vec<usize>, rewards: &[f64;41], leaving_penalty: &f64, compute_reward_cache: &mut ComputeRewardCache,
    position_value_cache: &mut PositionValueCache) -> (f64,usize,bool,Option<usize>) {
    /*
    Input:
    - Current score
    - picked_dice: dice values we already have
    - remaining_dice: number of dice left to throw
    - throw_result: the result of any given throw, given by an integer 0-3002 (8+6 choose 6 possible different throws of these dice)
    - list_throws: a vector to convert the throw_result index into a tuple, containing the number of each dice value in our throw
    - list_intervals: indexes giving the startpoints of intervals in list_throws, corresponding to 8,7,6,... dice thrown
    - rewards: vector of rewards
    - leaving_penalty: the penalty for failing a round
    
    Output:
    - The value of the state we are in
    - The best choice among the different dice values in the throw
    - Whether we have stopped playing or not
    - The best pickomino we can pick if we decide to stop playing.
    */    
    let mut max_reward = *leaving_penalty;
    let mut max_reward_choice = 100;
    let mut max_reward_continue_or_not = false;
    let mut pickomino_choice = None;
    let throw = list_throws[*throw_result];

    // Compute the values of each possible next state, as well as the value of quitting now.

    for dice in 0..6 {
        if picked_dice[dice] !=1 && throw[dice] !=0 {
            let new_score = *current_score + throw[dice]*std::cmp::min(5,dice+1);
            let mut new_picked_dice = picked_dice.clone();
            new_picked_dice[dice] = 1;
            let new_remaining_dice = remaining_dice - throw[dice];
            let x = position_value(&new_score, &new_picked_dice, &new_remaining_dice, list_throws, list_intervals, rewards, leaving_penalty, position_value_cache);          
            if x > max_reward {
                max_reward_choice = dice;
                max_reward = x;
                max_reward_continue_or_not = true;
            }
        }
    }

    //println!("max_reward {}", max_reward);
    (max_reward, max_reward_choice, max_reward_continue_or_not, pickomino_choice)
}

#[pyfunction]
fn compute_best_path_py(
    current_score: usize,
    picked_dice: Vec<usize>,  // Adjust types for Python compatibility
    remaining_dice: usize,
    throw_result: usize,
    rewards: Vec<f64>,  // Adjust types for Python compatibility
    leaving_penalty: f64,
) -> PyResult<(f64, usize, bool, Option<usize>)> {
    let mut compute_reward_cache: ComputeRewardCache = HashMap::new();
    let mut position_value_cache: PositionValueCache = HashMap::new();

    // Convert Python types to Rust types if necessary

    // Convert Python types (Vec) to Rust types ([usize; 6])
    let picked_dice_rust: [usize; 6] = picked_dice.try_into().map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("Expected a list of 6 dice values")
    })?;

    
    // Now call compute_best_path with all required arguments
    let result = compute_best_path(
        &current_score,
        &picked_dice_rust,
        &remaining_dice,
        &throw_result,
        &LIST_THROWS,
        &LIST_INTERVALS,
        &rewards.try_into().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Expected a list of 41 reward values")
        })?,
        &leaving_penalty,
        &mut compute_reward_cache,
        &mut position_value_cache,
    );

    Ok(result)
}

#[pymodule]
fn rust_library(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_best_path_py, m)?)?;
    m.add_function(wrap_pyfunction!(position_value_py, m)?)?;
    Ok(())
}
// Computes the binomial coefficient (n choose k)
fn binom(n: usize, k: usize) -> usize {
    if k > n {
        return 0;  // Return an appropriate value when k is greater than n
    }
    
    let mut result = 1;
    let mut k = k;
    
    if k > n - k {
        k = n - k;
    }
    
    for i in 0..k {
        result *= n - i;
        result /= i + 1;
    }
    
    result
}

fn index_partition() -> Vec<[usize; 6]> {
    let mut list_throws = Vec::new();
    for i_0 in 0..9 {
        for i_1 in 0..(9-i_0) {
            for i_2 in 0..(9-i_0-i_1) {
                for i_3 in 0..(9-i_0-i_1-i_2) {
                    for i_4 in 0..(9-i_0-i_1-i_2-i_3) {
                        for i_5 in 0..(9-i_0-i_1-i_2-i_3-i_4) {
                            let i_6 = 8-i_0-i_1-i_2-i_3-i_4-i_5;
                            list_throws.push([i_1 as usize, i_2 as usize, i_3 as usize, i_4 as usize, i_5 as usize, i_6 as usize]);
                        }
                    }
                }
            }
        }
    }
    list_throws
}

lazy_static! {
    static ref LIST_THROWS: Vec<[usize; 6]> = index_partition();
    static ref LIST_INTERVALS: Vec<usize> = {
        let list_points: Vec<usize> = (5..=13).rev().map(|i| binom(i, 5)).collect();
        let mut list_intervals = vec![0];
        for point in &list_points {
            let last = *list_intervals.last().unwrap();
            list_intervals.push(last + point);
        }
        list_intervals
    };
    static ref LIST_PROBABILITIES: Vec<f64> = compute_throws_probabilities(&LIST_THROWS, &LIST_INTERVALS);
}