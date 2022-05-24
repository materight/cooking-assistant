"""General actions utilities."""
from typing import Text, List, Optional
import numpy as np
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
    amount = None
    # If the unit is missing, assume minutes
    if unit is None:
        unit, original_unit = 'minutes', ''
    # Try to parse time amount
    try:
        amount = w2n.word_to_num(time_str.replace(original_unit, '').strip())
    except:
        amount, unit = None, None
    return amount, unit


def join_list_str(list_str: List[Text], last_sep: Text = 'and'):
    """Join a list of strings with commas and final "and"."""
    if len(list_str) == 0:
        return ''
    elif len(list_str) == 1:
        return list_str[0]
    else:
        return ', '.join(list_str[:-1]) + f' {last_sep} ' + list_str[-1]


def ingredient_to_str(name: Text, amount: Optional[Text], unit: Optional[Text], sep: Text = '', default_amount: Text = ''):
    """Convert the ingredient informations to a single string."""
    res = ''
    if amount is not None and not np.isnan(amount): 
        res += f'{amount:{ ".0f" if amount.is_integer() else ".1f"}}'
        if unit:
            res += f'{unit} {sep}'
        else:
            res += f' {sep}' if sep else ' x '
    elif default_amount:
        res += f'{default_amount} '
    res += name
    return res
