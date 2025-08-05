import json
import os
from collections import Counter
import re


def evaluation(input_path, **kwargs):
    """
    Evaluate NER results by comparing predicted entities with standard answers.
    Handles repeated entities - entities that appear multiple times in the standard
    answers must also appear the same number of times in predictions for full credit.
    
    Args:
        input_path (str): Path to the input jsonl file containing predictions and standard answers.
        **kwargs: Additional arguments (not used in this implementation).
        
    Returns:
        float: The overall accuracy score.
    """
    total_score = 0
    total_examples = 0

    output_path = os.path.join(os.path.dirname(input_path), "output.jsonl")
    
    with open(input_path, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
        for i,line in enumerate(f_in):
            example = json.loads(line.strip())

            prediction = example.get('predict_result', '')
            
            standard_answers = []
            for choice in example.get('choices', []):
                message = choice.get('message', {})
                content = message.get('content', [])
                
                for item in content:
                    if item.get('type') == 'text':
                        text = item.get('text', '')
                        try:
                            parsed = json.loads(text)
                            if isinstance(parsed, list):
                                for entity_group in parsed:
                                    for group_key, entities in entity_group.items():
                                        standard_answers.extend(entities)
                        except json.JSONDecodeError:
                            continue
            
            pred_entities = []
            try:
                matches = re.findall(r'"results":\s*"?(\[.*?\])', prediction, re.DOTALL)
                pred_entities = json.loads(matches[-1]) if matches else []
            except json.JSONDecodeError:
                pred_entities = []
            def entity_to_tuple(entity):
                if isinstance(entity, dict):
                    return (entity.get('type', ''), entity.get('text', ''))
                else:
                    return ('', '')  
            std_entity_counter = Counter(entity_to_tuple(entity) for entity in standard_answers)
            pred_entity_counter = Counter(entity_to_tuple(entity) for entity in pred_entities)
            hits = []
            missed = []
            extra = []
            
            for std_entity_tuple, std_count in std_entity_counter.items():
                pred_count = pred_entity_counter.get(std_entity_tuple, 0)

                std_entity_format = next((e for e in standard_answers if entity_to_tuple(e) == std_entity_tuple), {})
                
                match_count = min(std_count, pred_count)
                for _ in range(match_count):
                    hits.append(std_entity_format)
                
                for _ in range(std_count - match_count):
                    missed.append(std_entity_format)
            
            for pred_entity_tuple, pred_count in pred_entity_counter.items():
                
                std_count = std_entity_counter.get(pred_entity_tuple, 0)
                
                pred_entity_format = next((e for e in pred_entities if entity_to_tuple(e) == pred_entity_tuple), {})
                
                for _ in range(max(0, pred_count - std_count)):
                    extra.append(pred_entity_format)

            total_std_entities = len(standard_answers)
            total_hits = len(hits)
            score = total_hits / total_std_entities if total_std_entities > 0 else 0
            
            example['evalresult'] = {
                'hits': hits,
                'missed': missed, 
                'extra': extra,  
                'total_standard_entities': total_std_entities,
                'total_hits': total_hits,
                'score': score
            }

            f_out.write(json.dumps(example, ensure_ascii=False) + '\n')

            total_score += score
            total_examples += 1

    overall_score = total_score / total_examples if total_examples > 0 else 0
    return {"acc": overall_score}
