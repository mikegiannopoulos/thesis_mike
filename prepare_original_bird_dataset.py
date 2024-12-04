import pandas as pd

def load_and_clean_nestling_data(filename, nestlings=False):
    """
    ds: Raw data. All occurences in Sweden
    nestlings: If only nestling occurences needed turn True
    """
    ds = pd.read_csv(filename, sep="\t", on_bad_lines='skip')
    
    # Clean dataset
    ds.dropna(axis=1, how='all', inplace=True)
    
    col_to_drop1 = ['gbifID', 'occurrenceID', 'recordedBy', 
                    'organismQuantity', 'taxonomicStatus', 'eventTime',
                    'eventID', 'startDayOfYear', 'endDayOfYear',
                    'sampleSizeValue', 'locationID', 'county',
                    'taxonID', 'scientificName', 'order', 'family', 'genus', 'genericName',
                    'specificEpithet', 'infraspecificEpithet', 'taxonRank',
                    'vernacularName', 'lastInterpreted', 'taxonKey', 'acceptedTaxonKey',
                    'orderKey', 'familyKey', 'genusKey', 'speciesKey',
                    'acceptedScientificName', 'verbatimScientificName', 'lastParsed',
                    'level0Gid', 'level0Name', 'level1Gid', 'level1Name', 'level2Gid', 'year', 'month', 'day',
                    'level2Name', 'iucnRedListCategory']
    col_to_drop2 = [x for x in ds.columns if len(ds[x].unique()) == 1]
    ds.rename(columns={'decimalLatitude': 'lat', 'decimalLongitude': 'lon'}, inplace=True)
    
    ds = ds.drop(columns=col_to_drop1 + col_to_drop2)
    
    ds = ds[['eventDate', 'lifeStage', 'individualCount', 'lat', 'lon']]
    if nestlings:
        ds = ds[ds.lifeStage == 'Nestling']

    return ds


if __name__ == '__main__':
    filename = '/home/michael/masters_thesis/bird_data/occurrence.txt'
    ds = load_and_clean_nestling_data(filename, nestlings=False)
    
    ### Uncomment to save the csv
    ds.to_csv('/home/michael/masters_thesis/bird_data/all_nestlings_cleaned.csv')
