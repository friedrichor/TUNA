from ...smp import *
from .multiple_choice import extract_answer_from_item
import numpy as np
import re

FAIL_MSG = 'Failed to obtain answer via API.'

VISUAL_CHARACTERISTICS = [
    'Low-Dynamic',
    'High-Dynamic',
    'Multi-Subject',
    'Multi-Scene',
]

SKILLS = [
    'Camera State',
    'Background Scene',
    'Subject Action',
    'Object Attribute',
]

TASKS = [
    'Camera Motion',
    'Camera Transition',
    'Scene Description',
    'Scene Transition',
    'Action Recognition',
    'Action Sequence',
    'Action-Subject Matching',
    'Object Recognition',
    'Object Appearance',
    'Object Location',
]


def get_dimension_rating(data_path):
    data = load(data_path)

    char_rating = {k: {} for k in ['overall'] + VISUAL_CHARACTERISTICS}
    for char in ['overall'] + VISUAL_CHARACTERISTICS:
        char_rating[char] = {
            'overall': '',
            'skill': {k: [] for k in SKILLS},
            'task': {k: [] for k in TASKS}
        }

    for i in range(len(data)):
        skill = data.iloc[i]['skill']
        task = data.iloc[i]['task']

        char_rating['overall']['skill'][skill].append(data.iloc[i]['score'])
        char_rating['overall']['task'][task].append(data.iloc[i]['score'])

        characteristic = data.iloc[i]['visual_characteristic'].split(',')
        for cha in characteristic:
            char_rating[cha]['skill'][skill].append(data.iloc[i]['score'])
            char_rating[cha]['task'][task].append(data.iloc[i]['score'])

    for char in ['overall'] + VISUAL_CHARACTERISTICS:
        overall_res_dur = f'{np.mean([x for x in sum(char_rating[char]["skill"].values(), []) if x >= 0]):.3f}'
        char_rating[char]['overall'] = overall_res_dur

        for skill in SKILLS:
            skill_res_dur = f'{np.mean([x for x in char_rating[char]["skill"][skill] if x >= 0]):.3f}'
            char_rating[char]['skill'][skill] = skill_res_dur

        for task in TASKS:
            task_res_dur = f'{np.mean([x for x in char_rating[char]["task"][task] if x >= 0]):.3f}'
            char_rating[char]['task'][task] = task_res_dur

    return char_rating

def extract_option(model, input_item, dataset_name):
    options = input_item['question'].split('\n')[1:]
    for id, option in enumerate(options):
        option_id = chr(ord('A') + id) + '.'
        if option.find(option_id) >= 0:
            input_item[chr(ord('A') + id)] = option[option.find(option_id) + len(option_id):].strip('. \n')
    return extract_answer_from_item(model, input_item, dataset_name)['opt']


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        'The best answer is',
        'The correct answer is',
        'The answer is',
        'The answer',
        'The best option is'
        'The correct option is',
        'Best answer:'
        'Best option:',
        'Answer:',
        'Option:',
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, '')

    if len(s.split()) > 10 and not re.search('[ABCD]', s):
        return ''
    matches = re.search(r'[ABCD]', s)
    if matches is None:
        return ''
    return matches[0]


def build_prompt(item):
    tmpl = 'Question: {}\nGroundtruth answer: {}\nCandidate answer: {}\nYour response: '
    return tmpl.format(item['question'], item['answer'], item['prediction'])
