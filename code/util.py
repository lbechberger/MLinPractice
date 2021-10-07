#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility file for collecting frequently used constants and helper function.
"""

# column names for the original data frame
COLUMN_TWEET = "tweet"
COLUMN_LIKES = "likes_count"
COLUMN_RETWEETS = "retweets_count"

# column names of novel columns for preprocessing
COLUMN_LABEL = "label"
COLUMN_PUNCTUATION = "tweet_no_punctuation"

SUFFIX_TOKENIZED = "_tokenized"



def fm(given: str, should: "str"):
    """
    Formats the passed given and should as a string across two lines.
    Can be used for documenting assertions in tests.
    And helps understanding what went wrong in failed tests
    
    given: a sentence as a string,
    should: return a list of words as a string
    """
    return _format_message(given, should)


def _format_message(given: str, should: "str"):
    """
    See docstring of fm function
    Other example
    given: 'no arguments',
    should: 'return 0',
    """
    return f"\n given: {given},\n should: {should}"