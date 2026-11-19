"""Play-style tags for players and teams.

Every rule reads standardised numbers (0 = average at the position or in the league, +1 = about the top sixth)
that have already been steadied for sample size, so a tag needs a real body of work behind it. Each rule was
tested (research/traits_eval.py): a tag is shown only if players or teams that earned it through week 8 kept
showing it over the rest of the season clearly more often than chance. Rules that failed are listed in
research/traits_results.txt and are not shown.
"""
import json, os

KEEP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trait_rules.json')
# (name, plain-English tip, rule on the attribute z-scores)
PLAYER = {
    'QB': [('Playmaker', 'Adds a lot of expected points per play: well above the average QB.', lambda a: a['EFF'] >= 1.0),
           ('Dual-threat', 'Runs a lot: rushing yards per game well above the average QB.', lambda a: a['RUN'] >= 1.0),
           ('Gunslinger', 'Throws deep and takes risks: long average throw with more interceptions than most.', lambda a: a['DEEP'] >= 0.7 and a['SAFE'] <= -0.3),
           ('Game manager', 'Short, safe throws: low average depth and few interceptions.', lambda a: a['SAFE'] >= 0.5 and a['DEEP'] <= -0.7),
           ('Accurate', 'Completes more passes than expected given how hard his throws are.', lambda a: a['ACC'] >= 1.0),
           ('Deep passer', 'Average throw travels far downfield.', lambda a: a['DEEP'] >= 1.0),
           ('Pocket passer', 'Rarely runs and rarely takes sacks.', lambda a: a['RUN'] <= -0.7 and a['PKT'] >= 0.3),
           ('Sack-prone', 'Takes sacks on a large share of his dropbacks.', lambda a: a['PKT'] <= -1.0),
           ('Ball-safe', 'Very few interceptions per attempt.', lambda a: a['SAFE'] >= 1.0)],
    'RB': [('Workhorse', 'Carries the load: carries per game well above the average back.', lambda a: a['WORK'] >= 1.0),
           ('Big-play back', 'A large share of his carries gain 10+ yards.', lambda a: a['BIG'] >= 1.0),
           ('Receiving back', 'Used as a pass catcher: receiving yards per game well above the average back.', lambda a: a['REC'] >= 1.0),
           ('Efficient', 'Adds more expected points per touch than most backs.', lambda a: a['EFF'] >= 1.0),
           ('Goal-line back', 'Scores often without many long runs.', lambda a: a['TD'] >= 1.0 and a['BIG'] <= 0.0)],
    'WR': [('Target hog', 'Gets a large share of his team\'s targets.', lambda a: a['TGT'] >= 1.0),
           ('Deep threat', 'Targeted far downfield: average target depth well above the position.', lambda a: a['DEEP'] >= 1.0),
           ('Possession receiver', 'Reliable on shorter throws: high catch rate, short average target.', lambda a: a['HNDS'] >= 0.7 and a['DEEP'] <= -0.5),
           ('YAC threat', 'Gains a lot after the catch.', lambda a: a['YAC'] >= 1.0),
           ('Red-zone threat', 'Scores touchdowns at a high rate.', lambda a: a['TD'] >= 1.0),
           ('Efficient', 'Adds more expected points per target than most.', lambda a: a['EFF'] >= 1.0),
           ('Sure hands', 'Catches a high share of his targets.', lambda a: a['HNDS'] >= 1.0)],
    'EDGE': [('Pass rusher', 'Gets sacks at a high rate.', lambda a: a['SACK'] >= 1.0),
             ('Pressure player', 'Hits the QB often even when he does not get the sack.', lambda a: a['PRES'] >= 1.0),
             ('Run stopper', 'Ends running plays for little gain, without big sack numbers.', lambda a: a['STOP'] >= 1.0 and a['SACK'] < 0.5),
             ('Backfield disruptor', 'Tackles runners and receivers behind the line.', lambda a: a['TFL'] >= 1.0),
             ('Strip specialist', 'Forces fumbles at a high rate.', lambda a: a['STRP'] >= 1.0)],
    'LB': [('Tackling machine', 'Makes tackles everywhere: tackles per game well above the average linebacker.', lambda a: a['TKL'] >= 1.0),
           ('Run stopper', 'Ends plays for little gain or behind the line.', lambda a: a['STOP'] >= 1.0 or a['TFL'] >= 1.0),
           ('Blitzer', 'Gets to the QB: sacks and QB hits well above the average linebacker.', lambda a: a['RUSH'] >= 1.0),
           ('Coverage LB', 'Breaks up passes at a high rate.', lambda a: a['COV'] >= 1.0),
           ('Takeaway artist', 'Interceptions and forced fumbles at a high rate.', lambda a: a['BALL'] >= 1.0)],
    'CB': [('Sticky coverage', 'Breaks up passes at a high rate.', lambda a: a['COV'] >= 1.0),
           ('Ball hawk', 'Intercepts passes at a high rate.', lambda a: a['BALL'] >= 1.0),
           ('Run support', 'Comes up and ends plays for little gain.', lambda a: a['STOP'] >= 1.0),
           ('Strip specialist', 'Forces fumbles at a high rate.', lambda a: a['STRP'] >= 1.0)],
    'K': [('Big leg', 'Makes more long kicks than expected.', lambda a: a['LONG'] >= 1.0),
          ('Automatic', 'Makes more field goals than expected for their distance.', lambda a: a['ACC'] >= 1.0)],
}
PLAYER['TE'], PLAYER['DL'] = PLAYER['WR'], PLAYER['EDGE']
PLAYER['S'] = [(('Box safety' if n == 'Run support' else n), t, f) for n, t, f in PLAYER['CB']]
TEAM = [('Pass-first offense', 'Throws more often than the situation calls for.', lambda a: a['proe'] >= 1.0),
        ('Run-first offense', 'Runs more often than the situation calls for.', lambda a: a['proe'] <= -1.0),
        ('Strong passing game', 'Dropbacks produce far more than average, after adjusting for the defenses faced.', lambda a: a['o_drop'] >= 1.0),
        ('Strong run game', 'Runs produce far more than average, after adjusting for the defenses faced.', lambda a: a['o_rush'] >= 1.0),
        ('Explosive offense', 'Lots of 20+ yard plays.', lambda a: a['o_expl'] >= 1.0),
        ('Methodical offense', 'Stays on schedule with steady gains but few big plays.', lambda a: a['o_succ'] >= 0.7 and a['o_expl'] <= 0.0),
        ('Turnover-prone', 'Loses a lot through interceptions.', lambda a: a['o_int'] <= -1.0),
        ('Pass-rush defense', 'Sacks cost opponents far more than average.', lambda a: a['d_sack'] >= 1.0),
        ('No-fly zone', 'Opponents\' dropbacks produce far less than average.', lambda a: a['d_drop'] >= 1.0),
        ('Run wall', 'Opponents\' runs produce far less than average.', lambda a: a['d_rush'] >= 1.0),
        ('Opportunistic defense', 'Takes a lot away through interceptions.', lambda a: a['d_int'] >= 1.0),
        ('Big-play prone defense', 'Gives up a lot of 20+ yard plays.', lambda a: a['d_expl'] <= -1.0),
        ('Leaky pass defense', 'Opponents\' dropbacks produce far more than average.', lambda a: a['d_drop'] <= -1.0),
        ('Disciplined', 'Penalties help this team far more than they hurt it.', lambda a: a['n_pen'] >= 1.0),
        ('Penalty-prone', 'Penalties cost this team far more than average.', lambda a: a['n_pen'] <= -1.0),
        ('Strong special teams', 'Kicking, punting and returns add points.', lambda a: a['o_st'] >= 1.0)]
_KEEP = None


def keep():
    global _KEEP
    if _KEEP is None:
        _KEEP = json.load(open(KEEP_PATH)) if os.path.exists(KEEP_PATH) else {}
    return _KEEP


def player(pos, a, params=None, everything=False, limit=2):
    ok = keep().get(pos)
    out = []
    for name, tip, rule in PLAYER.get(pos, []):
        try:
            if (everything or ok is None or name in ok) and rule(a):
                out.append([name, tip])
        except KeyError:
            pass
    return out if everything else out[:limit]


def team(a, everything=False, limit=4):
    ok = keep().get('TEAM')
    out = [[n, t] for n, t, rule in TEAM if (everything or ok is None or n in ok) and rule(a)]
    return out if everything else out[:limit]
