"""Independent manufactured JRA field specification shared by input controls."""
# Explicit fixture specification; deliberately does not import the reader's FIELDS.
WEATHER = [
    ('wind_u', 'uas', 'm s-1', 6., 'point', 10800, 10),
    ('wind_v', 'vas', 'm s-1', 2., 'point', 10800, 10),
    ('temperature_k', 'tas', 'K', 290., 'point', 10800, 10),
    ('specific_humidity', 'huss', '1', .005, 'point', 10800, 10),
    ('pressure_pa', 'psl', 'Pa', 101325., 'point', 10800, None),
    ('shortwave_down', 'rsds', 'W m-2', 100., 'mean', 10800, None),
    ('longwave_down', 'rlds', 'W m-2', 320., 'mean', 10800, None),
    ('rain', 'prra', 'kg m-2 s-1', 1e-5, 'mean', 10800, None),
    ('snow', 'prsn', 'kg m-2 s-1', 0., 'mean', 10800, None),
    ('runoff', 'friver', 'kg m-2 s-1', 1e-5, 'mean', 86400, None),
    ('calving', 'licalvf', 'kg m-2 s-1', 0., 'mean', 86400, None),
]

