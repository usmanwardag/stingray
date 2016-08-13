
def load_events(mission):
	"""Load column names and corresponding extensions for fits event 
	files of different missions.

	Parameter
	---------
	mission: str
		Name of the mission. Accepts 'RXTE', 'XMM' and 'NuSTAR'
	"""
	
	config = {'RXTE': {'TIME':'EVENTS', 'PI':'EVENTS', 'TIMEZERO':'EVENTS', 'MJDREF':'EVENTS',
					'TSTART':'EVENTS', 'TSTOP':'EVENTS', 'START':'SGTI', 'STOP':'SGTI'},
			'XMM': {'TIME':'EVENTS', 'PI':'EVENTS', 'TIMEZERO':'EVENTS', 'MJDREF':'EVENTS',
					'TSTART':'EVENTS', 'TSTOP':'EVENTS', 'START':'GTI', 'STOP':'GTI'},
			'NuSTAR': {'TIME':'EVENTS', 'PI':'EVENTS', 'TIMEZERO':'EVENTS', 'MJDREF':'EVENTS',
					'TSTART':'EVENTS', 'TSTOP':'EVENTS', 'START':'GTI', 'STOP':'GTI'}}

	return config[mission]

def load_lcurve(mission):
	"""Load column names and corresponding extensions for fits lightcurve 
	files of different missions.

	Parameter
	---------
	mission: str
		Name of the mission. Accepts 'RXTE', 'XMM' and 'NuSTAR'
	"""
	
	config = {'RXTE': ['','','','',''],
			'XMM': ['','','','',''],
			'NuSTAR': ['','','','','']}

	return config[mission]

def load_spectrum(mission):
	"""Load column names and corresponding extensions for fits spectrum file
	of different missions.

	Parameter
	---------
	mission: str
		Name of the mission. Accepts 'RXTE', 'XMM' and 'NuSTAR'
	"""

	config = {'RXTE': ['','','','',''],
			'XMM': ['','','','',''],
			'NuSTAR': ['','','','','']}

	return config[mission]