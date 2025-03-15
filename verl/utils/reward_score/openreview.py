# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import math

def format_reward(predict_str: str) -> float:
    """
    Check if the prediction follows the format <think>...</think><answer>X</answer>
    where X is an integer between 1 and 10.
    """
    pattern = re.compile(r'^<think>.*</think>[\s\n]*<answer>([1-9]|10)</answer>$', re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0

def extract_answer(predict_str: str) -> int:
    """
    Extract the integer rating from the <answer> tags.
    """
    pattern = re.compile(r'<answer>([1-9]|10)</answer>', re.DOTALL)
    match = re.search(pattern, predict_str)
    if match:
        return int(match.group(1))
    return 0
    

def answer_reward(predict_str: str, ground_truth: str) -> float:
    """
    Calculate MSE loss between predicted answer and ground truth.
    Lower MSE means higher reward.
    """
    try:
        predicted_value = extract_answer(predict_str)
        ground_truth_value = float(ground_truth)
        
        if predicted_value == 0:  # Failed to extract answer
            return 0.0, 0.0, 0.0
            
        mae = float(abs(predicted_value - ground_truth_value))
        max_possible_mae = 9  # Maximum possible MAE: (10-1)
        max_possible_mse = 81  # Maximum possible MSE: (10-1)^2

        # normalize reward between 0 and 1 (higher is better)
        answer_mae_reward = 1.0 - (mae / max_possible_mae) # 0 is the worst, 1 is the best
        answer_mse_reward = 1.0 - (mae / max_possible_mse) # 0 is the worst, 1 is the best

        answer_exact_reward = 1.0 if int(predicted_value) == int(ground_truth_value + 0.5) else 0.0

        return answer_mae_reward, answer_mse_reward, answer_exact_reward
    except:
        return 0.0, 0.0, 0.0
    
def compute_score(predict_str: str, ground_truth: str) -> float:
    format_rew = format_reward(predict_str)
    ans_mae_rew, ans_mse_rew, ans_exact_rew = answer_reward(predict_str, ground_truth)
    total_score = 1.0 * format_rew + 1.0 * ans_mae_rew + 3.0 * ans_exact_rew
    
    # Print detailed information for each sample
    predicted_value = extract_answer(predict_str)
    print("-" * 50)
    print(f"Sample Details:")
    print(f"  Prediction string: {predict_str}")  # Show first 100 chars
    print(f"  Extracted answer: {predicted_value}")
    print(f"  Ground truth: {ground_truth}")
    print(f"  Format reward: {format_rew}")
    print(f"  Answer reward: mae {ans_mae_rew}, mse {ans_mse_rew}, exact {ans_exact_rew}")
    print(f"  Total score: {total_score}")
    print("-" * 50)
    
    return total_score

def compute_test_score(predict_str: str, ground_truth: str) -> float:
    """
    Compute the overall score based on the negative MAE (Mean Absolute Error).
    """
    predicted_value = extract_answer(predict_str)
    if predicted_value == 0:
        predicted_value = 5 # when computing score, if the answer is not extracted, set it to 5 (the middle value)
    ground_truth_value = float(ground_truth)
    mae_score = - float(abs(predicted_value - ground_truth_value))
    
    # Print detailed information for test samples
    print("-" * 50)
    print(f"Test Sample Details:")
    print(f"  Prediction string: {predict_str}")  # Show first 100 chars
    print(f"  Extracted answer: {predicted_value}")
    print(f"  Ground truth: {ground_truth}")
    print(f"  Negative MAE score: {mae_score}")
    print("-" * 50)
    
    return mae_score
