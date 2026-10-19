# Keys are nflverse abbreviations, which FootballDB's player tables also use.
# Values: full name (FootballDB team tables), Pro Football Reference abbreviation (honors list).
TEAMS = {
    'ARI': ('Arizona Cardinals', 'ARI'), 'ATL': ('Atlanta Falcons', 'ATL'),
    'BAL': ('Baltimore Ravens', 'BAL'), 'BUF': ('Buffalo Bills', 'BUF'),
    'CAR': ('Carolina Panthers', 'CAR'), 'CHI': ('Chicago Bears', 'CHI'),
    'CIN': ('Cincinnati Bengals', 'CIN'), 'CLE': ('Cleveland Browns', 'CLE'),
    'DAL': ('Dallas Cowboys', 'DAL'), 'DEN': ('Denver Broncos', 'DEN'),
    'DET': ('Detroit Lions', 'DET'), 'GB': ('Green Bay Packers', 'GNB'),
    'HOU': ('Houston Texans', 'HOU'), 'IND': ('Indianapolis Colts', 'IND'),
    'JAX': ('Jacksonville Jaguars', 'JAX'), 'KC': ('Kansas City Chiefs', 'KAN'),
    'LAC': ('Los Angeles Chargers', 'LAC'), 'LA': ('Los Angeles Rams', 'LAR'),
    'LV': ('Las Vegas Raiders', 'LVR'), 'MIA': ('Miami Dolphins', 'MIA'),
    'MIN': ('Minnesota Vikings', 'MIN'), 'NE': ('New England Patriots', 'NWE'),
    'NO': ('New Orleans Saints', 'NOR'), 'NYG': ('New York Giants', 'NYG'),
    'NYJ': ('New York Jets', 'NYJ'), 'PHI': ('Philadelphia Eagles', 'PHI'),
    'PIT': ('Pittsburgh Steelers', 'PIT'), 'SF': ('San Francisco 49ers', 'SFO'),
    'SEA': ('Seattle Seahawks', 'SEA'), 'TB': ('Tampa Bay Buccaneers', 'TAM'),
    'TEN': ('Tennessee Titans', 'TEN'), 'WAS': ('Washington Commanders', 'WAS'),
}
ABBRS = sorted(TEAMS)
IDX = {t: i for i, t in enumerate(ABBRS)}
BY_NAME = {name: ab for ab, (name, _) in TEAMS.items()}
BY_PFR = {pfr: ab for ab, (_, pfr) in TEAMS.items()}


def name(ab):
    return TEAMS[ab][0]


def display_abbr(ab):
    return 'LAR' if ab == 'LA' else ab
