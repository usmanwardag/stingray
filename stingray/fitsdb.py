
def load_events():
	"""Load column names and corresponding extensions for fits event 
	files of different missions.
	"""
	
	config = {'RXTE': [{'TIME':'EVENTS'}, {'PI':'EVENTS'}, {'TIMEZERO':'EVENTS'}, {'MJDREF':'EVENTS'},
					{'TSTART':'EVENTS'}, {'TSTOP':'EVENTS'}, {'START':'SGTI'}, {'STOP':'SGTI'}]
			'XMM': [{'TIME':'EVENTS'}, {'PI':'EVENTS'}, {'TIMEZERO':'EVENTS'}, {'MJDREF':'EVENTS'},
					{'TSTART':'EVENTS'}, {'TSTOP':'EVENTS'}, {'START':'GTI'}, {'STOP':'GTI'}],
			'NuSTAR': [{'TIME':'EVENTS'}, {'PI':'EVENTS'}, {'TIMEZERO':'EVENTS'}, {'MJDREF':'EVENTS'},
					{'TSTART':'EVENTS'}, {'TSTOP':'EVENTS'}, {'START':'GTI'}, {'STOP':'GTI'}]}

	return config

def load_lcurve():
	"""Load column names and corresponding extensions for fits lightcurve 
	files of different missions.
	"""
	
	config = {'RXTE': ['','','','',''],
			'XMM': ['','','','',''],
			'NuSTAR': ['','','','','']}

	return config

def load_spectrum():
	"""Load column names and corresponding extensions for fits spectrum file
	of different missions.
	"""
	
	config = {'RXTE': ['','','','',''],
			'XMM': ['','','','',''],
			'NuSTAR': ['','','','','']}

	return config