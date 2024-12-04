configs = {
    'air_temperature': {
        'file_path_pattern': '/home/michael/Education/UoG/Earth Science Master/Thesis/data/SMHI/meteorologi/air_temperature/*.csv',
        'skiprows': 10,
        'usecols': [0, 2],
        'delimiter': ';',
        'column_names': ['Datum', 'Lufttemperatur'],
        'date_column': 'Datum',
        'value_column': 'Lufttemperatur',
        'metrics': {
            'mean': {
                'monthly_metric_name': 'Monthly Mean Air Temperature',
                'yearly_metric_name': 'Yearly Mean Air Temperature'
            },
            'std': {
                'monthly_metric_name': 'Monthly Std Dev Air Temperature',
                'yearly_metric_name': 'Yearly Std Dev Air Temperature'
            }
        }
    },

    'air_pressure': {
        'file_path_pattern': '/home/michael/Education/UoG/Earth Science Master/Thesis/data/SMHI/meteorologi/air_pressure/*.csv',
        'skiprows': 9,
        'usecols': [0, 2],
        'delimiter': ';',
        'column_names': ['Datum', 'Lufttryck reducerat havsytans nivå'],
        'date_column': 'Datum',
        'value_column': 'Lufttryck reducerat havsytans nivå',
        'metrics': {
            'mean': {
                'monthly_metric_name': 'Monthly Mean Air Pressure',
                'yearly_metric_name': 'Yearly Mean Air Pressure'
            },
            'std': {
                'monthly_metric_name': 'Monthly Std Dev Air Pressure',
                'yearly_metric_name': 'Yearly Std Dev Air Pressure'
            },
        },
    },
    
    'sea_temperature': {
        'file_path_pattern': '/home/michael/Education/UoG/Earth Science Master/Thesis/data/SMHI/oceanografi/sea_temp/*.csv',
        'skiprows': 6,
        'usecols': [0, 1],
        'delimiter': ';',
        'column_names': ['Datum Tid (UTC)', 'Havstemperatur'],
        'date_column': 'Datum Tid (UTC)',
        'value_column': 'Havstemperatur',
        'metrics': {
            'mean': {
                'monthly_metric_name': 'Monthly Mean Sea Temperature',
                'yearly_metric_name': 'Yearly Mean Sea Temperature'
            },
            'std': {
                'monthly_metric_name': 'Monthly Std Dev Sea Temperature',
                'yearly_metric_name': 'Yearly Std Dev Sea Temperature'
            },
        },
    },
    
    'sea_water_level': {
        'file_path_pattern': '/home/michael/Education/UoG/Earth Science Master/Thesis/data/SMHI/oceanografi/sea_water_level/*.csv',
        'skiprows': 6,
        'usecols': [0, 1],
        'delimiter': ';',
        'column_names': ['Datum Tid (UTC)', 'Havsvattenstånd'],
        'date_column': 'Datum Tid (UTC)',
        'value_column': 'Havsvattenstånd',
        'metrics': {
            'mean': {
                'monthly_metric_name': 'Monthly Mean Sea Water Level',
                'yearly_metric_name': 'Yearly Mean Sea Water Level'
            },
            'std': {
                'monthly_metric_name': 'Monthly Std Dev Sea Water Level',
                'yearly_metric_name': 'Yearly Std Dev Sea Water Level'
            },
        },
    },
        
    'wave_height': {
        'file_path_pattern': '/home/michael/Education/UoG/Earth Science Master/Thesis/data/SMHI/oceanografi/wave_height/*.csv',
        'skiprows': 10,
        'usecols': [0, 1],
        'delimiter': ';',
        'column_names': ['Datum Tid (UTC)', 'Våghöjd'],
        'date_column': 'Datum Tid (UTC)',
        'value_column': 'Våghöjd',
        'metrics': {
            'mean': {
                'monthly_metric_name': 'Monthly Mean Wave Height',
                'yearly_metric_name': 'Yearly Mean Wave Height'
            },
            'std': {
                'monthly_metric_name': 'Monthly Std Dev Wave Height',
                'yearly_metric_name': 'Yearly Std Dev Wave Height'
            },
        },
    },
        
    'wind_speed': {
        'file_path_pattern': '/home/michael/Education/UoG/Earth Science Master/Thesis/data/SMHI/meteorologi/wind/*.csv',
        'skiprows': 10,
        'usecols': [0, 4],
        'delimiter': r'[;,]',
        'column_names': ['Datum', 'Vindhastighet'],
        'date_column': 'Datum',
        'value_column': 'Vindhastighet',
        'metrics': {
            'mean': {
                'monthly_metric_name': 'Monthly Mean Wind Speed',
                'yearly_metric_name': 'Yearly Mean Wind Speed'
            },
            'std': {
                'monthly_metric_name': 'Monthly Std Dev Wind Speed',
                'yearly_metric_name': 'Yearly Std Dev Wind Speed'
            },
        },
    },      
    # Add other variables with similar structure
}  
