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
    Check if the prediction follows the required format.
    
    Format requirements:
    - Must start with <think> tag followed by any content and closed with </think>
    - Must contain <answer>X</answer> where X is an integer from 1 to 10
    - May optionally end with <|im_end|> token
    - Whitespace between sections is allowed
    
    Args:
        predict_str: The string to check
        
    Returns:
        1.0 if the format is valid, 0.0 otherwise
    """
    # Compile regex pattern once at module level for better performance
    THINK_PART = r'<think>[\s\S]*</think>'
    ANSWER_PART = r'<answer>([1-9]|10)</answer>'
    END_PART = r'(?:<\|im_end\|>)?'
    VALID_FORMAT_PATTERN = re.compile(
        rf'^[\s\n]*{THINK_PART}[\s\n]*{ANSWER_PART}[\s\n]*{END_PART}[\s\n]*$', 
        re.DOTALL
    )
    match_result = re.fullmatch(VALID_FORMAT_PATTERN, predict_str)
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

        ground_truth_value_int = int(ground_truth_value + 0.5)
        answer_exact_reward = 1.0 if int(predicted_value) == ground_truth_value_int else 0.0

        """
        # use this when facing unbalance data
        answer_exact_reward_factor = {
            6: 0.05,
            5: 0.3,
            7: 0.3
        }
        if ground_truth_value_int in answer_exact_reward_factor:
            answer_exact_reward *= answer_exact_reward_factor[ground_truth_value_int]
        """

        return answer_mae_reward, answer_mse_reward, answer_exact_reward
    except:
        return 0.0, 0.0, 0.0
    
def compute_score(predict_str: str, ground_truth: str) -> float:
    format_rew = format_reward(predict_str)
    ans_mae_rew, ans_mse_rew, ans_exact_rew = answer_reward(predict_str, ground_truth)
    total_score = 1.0 * format_rew + 3.0 * ans_mae_rew + 1.0 * ans_exact_rew
    
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
