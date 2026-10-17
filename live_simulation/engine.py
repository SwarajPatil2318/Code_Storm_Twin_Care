"""Pure reproducible synthetic physiology, independent of trained artifacts."""
from datetime import timedelta
import hashlib
import math
from patient_connections.contracts import clock

PRESETS = {
    'stable': {'baseline': 112., 'circadian_amplitude': 4., 'meal_response': .55, 'recovery': .32},
    'moderate': {'baseline': 125., 'circadian_amplitude': 6., 'meal_response': .9, 'recovery': .32},
    'strong': {'baseline': 138., 'circadian_amplitude': 8., 'meal_response': 1.3, 'recovery': .32},
}
BOUNDS = {'glucose_mg_dl': [65., 350.], 'change_per_15_minutes_mg_dl': 18., 'heart_rate_bpm': [48., 145.]}


def noise(seed, step, channel):
    raw = hashlib.sha256(f'{seed}:{step}:{channel}'.encode()).digest()
    return int.from_bytes(raw[:8], 'big') / (2**64 - 1) * 2 - 1


def initial(at, profile):
    return {'time': at.isoformat(), 'step': 0, 'glucose': PRESETS[profile]['baseline'],
            'slope': 0., 'heart_rate': 68., 'steps_today': 0, 'active_minutes_today': 0,
            'activity': 'Resting', 'meal_effects': [], 'meal_revisions': {}}


def meal_effect(meal):
    # Reviewed estimates select broad synthetic input parameters, not a glucose prediction.
    carbs = meal.get('estimated_carb_range_g')
    load = (carbs['low'] + carbs['high']) / 2 if carbs else {
        'LOW': 15., 'MODERATE': 45., 'HIGH': 85., 'UNKNOWN': 0.}[meal['carbohydrate_category']]
    gi = meal.get('estimated_gi_range')
    gi_scale = ((gi['low'] + gi['high']) / 100) if gi else {
        'LOW': .65, 'MODERATE': 1., 'HIGH': 1.25, 'UNKNOWN': .8}[meal['glycemic_impact_category']]
    gl = meal.get('glycemic_load_category', 'UNKNOWN')
    gl_scale = {'LOW': .65, 'MODERATE': 1., 'HIGH': 1.25, 'UNKNOWN': 1.}[gl]
    return {'meal_id': meal['meal_id'], 'revision': meal['revision'], 'time': meal['meal_occurred_at'],
            'load': min(140., load), 'impact': gi_scale * gl_scale, 'peak_minutes': 60., 'duration_minutes': 360.}


def register_meals(state, meals, reference):
    """Only confirmed, available, already occurred meals enter future state evolution."""
    active_ids = set()
    for meal in meals:
        if meal['status'] != 'CONFIRMED' or not meal['user_confirmed']:
            continue
        if clock(meal['updated_at']) > reference or clock(meal['meal_occurred_at']) > reference:
            continue
        active_ids.add(meal['meal_id'])
        if state['meal_revisions'].get(meal['meal_id']) == meal['revision']:
            continue
        state['meal_effects'] = [m for m in state['meal_effects'] if m['meal_id'] != meal['meal_id']]
        state['meal_effects'].append(meal_effect(meal))
        state['meal_revisions'][meal['meal_id']] = meal['revision']
    state['meal_effects'] = [m for m in state['meal_effects'] if m['meal_id'] in active_ids
                            and (reference - clock(m['time'])).total_seconds() / 60 < m['duration_minutes']]
    # Bounded runtime state; expired meals are not reintroduced unless their revision changes.
    state['meal_revisions'] = {m['meal_id']: m['revision'] for m in meals
                              if clock(m['meal_occurred_at']) >= reference - timedelta(hours=8)}


def evolve(state, profile, seed, activity=None):
    """Native 15-minute recurrence, smooth meal targets, bounded momentum and noise."""
    result = {**state, 'meal_effects': [dict(m) for m in state['meal_effects']],
              'meal_revisions': dict(state['meal_revisions'])}
    at = clock(state['time']) + timedelta(minutes=15)
    step = state['step'] + 1
    preset = PRESETS[profile]
    hour = at.hour + at.minute / 60
    level = activity or state['activity']
    if level not in ('Resting', 'Walking'):
        raise ValueError('Choose Resting or Walking')
    meal_target = 0.
    for meal in state['meal_effects']:
        age = (at - clock(meal['time'])).total_seconds() / 60
        if 0 < age < meal['duration_minutes']:
            u = age / meal['peak_minutes']
            meal_target += meal['load'] * .65 * preset['meal_response'] * meal['impact'] * u * math.exp(1 - u)
    target = preset['baseline'] + preset['circadian_amplitude'] * math.sin((hour - 5) * math.pi / 12) + meal_target
    change = preset['recovery'] * (target - state['glucose']) + .12 * state['slope']
    change += -.9 if level == 'Walking' else 0.
    change += noise(seed, step, 'glucose') * .6
    change = max(-18., min(18., change))
    glucose = max(65., min(350., state['glucose'] + change))
    hr_target = 103. if level == 'Walking' else 62. if hour < 6 else 70.
    hr = max(48., min(145., state['heart_rate'] + .45 * (hr_target - state['heart_rate']) + noise(seed, step, 'hr')))
    reset_day = at.date() != clock(state['time']).date()
    steps = (0 if reset_day else state['steps_today']) + (1100 + int(noise(seed, step, 'steps') * 100) if level == 'Walking' else 20)
    active = (0 if reset_day else state['active_minutes_today']) + (15 if level == 'Walking' else 0)
    result.update(time=at.isoformat(), step=step, glucose=round(glucose, 3), slope=round(glucose - state['glucose'], 3),
                  heart_rate=round(hr, 1), steps_today=steps, active_minutes_today=active, activity=level)
    result['meal_effects'] = [m for m in result['meal_effects'] if (at - clock(m['time'])).total_seconds() / 60 < m['duration_minutes']]
    return result
