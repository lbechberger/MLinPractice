#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Periodize the daytime into night, morning, afternoon and evening

Created on Thu Oct  28 17:35:24 2021

@author: chbroecker
"""


class DayPeriodizer():
    # Default values are:
    # 0  - 6  -> night
    # 6  - 12 -> morning 
    # 12 - 18 -> afternoon
    # 18 - 0  -> evening
    def __init__(self, night_end, morning_end, afternoon_end, evening_end):
        self.night_end = night_end
        self.morning_end = morning_end
        self.afternoon_end = afternoon_end
        self.evening_end = evening_end

    def day_periodize(self, time):
        hour = int(time.split(':')[0])
        if hour < self.night_end:
            return 'Night'
        elif hour < self.morning_end:
            return 'Morning'
        elif hour < self.afternoon_end:
            return 'Afternoon'
        elif hour <= self.evening_end:
            return 'Evening'