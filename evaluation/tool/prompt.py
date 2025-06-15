PROMPT_SPLIT_EVENT = """
Given a chronological video caption, split it into multiple chronologically evolving events.
All events spliced together should be equal to the original caption.

Video Caption:
{caption}

Output a List event_list formed as:
[event1, event2, ...]
where ```video_caption = ' '.join(event_list)```
Note: If there are lots of repeats, please delete the repeats.

DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only output the List. Output:
"""

PROMPT_MATCH_EVENT = """
Given a chronological list of candidate events and a chronological list of reference events.
Finds a matching reference event for each candidate event and returns a tuple of ids (candidate_id, reference_id) for both. If there is no match then reference_id is None.
Note that event matching should also be done in chronological order.
Each reference event can be matched by multiple candidate events.

Candidate Events:
{candidate_events}
Reference Events:
{reference_events}

Output a List formed as:
[(1, reference_id_1), (2, reference_id_2), ...]
where, reference_id_1 <= reference_id_2 <= ... <= reference_id_n if reference_id_i is not None.

DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only output the List. Output:
"""

PROMPT_CLASSIFY_RELATION = """
Given a series of candidate events and corresponding ground-truth visual elements contained in a video, you need to judge whether the candidate event accurately and completely describes each visual elements.
For each event, classify the relationship between the candidate event and the ground-truth visual elements into three classes: entailment, lack, contradiction.
- "entailment": the candidate event entails the visual element.
- "lack": the candidate event lacks the visual element.
- "contradiction": some detail in the candidate event contradicts with the visual element. Pay attention to the correspondence between the character and the action.

Candidate Events and Ground-truth Visual Elements:
{match_data}

Output a JSON formed as:
[
  {{
    "candidate_event": "copy the candidate_event here",
    "visual_elements": [{{"content": "copy the visual_element here", "relationship": "put class name here"}}, ... ]
  }},
  ...
]

DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only output the JSON. Output:
"""
