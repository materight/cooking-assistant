"""General actions utilities."""
from typing import Text

from word2number import w2n


UNIT_MAPPINGS = dict(
    hours=['hours', 'hour'],
    minutes=['minutes', 'minute', 'min'],
    seconds=['seconds', 'second', 'sec']
)


def lower_first_letter(text: Text):
    """Lower the first letter of a string."""
    return text[0].lower() + text[1:]


def parse_time_unit(unit_str: Text):
    """Parse a unit string to a standard time unit."""   
    for key, units in UNIT_MAPPINGS.items():
        for unit in units:
            if unit in unit_str:
                return key, unit
    return None, None

def parse_time_str(time_str: Text):
    """Parse a time string into the corresponding time amount and unit."""
    unit, original_unit = parse_time_unit(time_str)
    if unit is not None and original_unit is not None:
        try:
            amount = w2n.word_to_num(time_str.replace(original_unit, ''))
        except:
            amount = None
    return amount, unit
