import os
import glob
import json
from config import Config
from filemanager import FileManager
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import time
import tqdm

def get_country(location):
    geolocator = Nominatim(user_agent="my_agent")
    try:
        # geopyに直接locationを渡して解析を試みる
        geo_location = geolocator.geocode(location, language='en')
        if geo_location:
            # 国名を取得
            address = geolocator.reverse(f"{geo_location.latitude}, {geo_location.longitude}", language='en')
            if address and 'country' in address.raw['address']:
                return address.raw['address']['country']
    except (GeocoderTimedOut, GeocoderUnavailable):
        pass
    return None

# リクエスト制限を回避するための関数
def rate_limited(max_per_second):
    min_interval = 1.0 / float(max_per_second)
    def decorate(func):
        last_time_called = [0.0]
        def rate_limited_function(*args, **kwargs):
            elapsed = time.perf_counter() - last_time_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_time_called[0] = time.perf_counter()
            return ret
        return rate_limited_function
    return decorate

# get_country関数をレート制限付きで再定義
@rate_limited(1)
def rate_limited_get_country(location):
    return get_country(location)



filemanager = FileManager(Config)

target_key = 'EMBERS___Geographic Location (Latitude and Longitude)'

if not os.path.exists('./geo_loc_dict.json'):
    all_geoloc_values = set()
    for pmc_dir in glob.glob(os.path.join(Config.DATA_DIR, 'PMC*')):
        pmc_number = pmc_dir.split('/')[-1]
        if not filemanager.check_if_samples_json_exists(pmc_number):
            continue
        samples = filemanager.load_samples_json(pmc_number)

        aligned_values = [s[target_key]['Aligned'] for s in samples if s.get(target_key) and s.get(target_key).get('Aligned')]
        aligned_values_set = set(aligned_values)
        if len(aligned_values_set) == 0:
            continue
        else:
            all_geoloc_values.update(aligned_values_set)

    print('Total number of unique geoloc values:', len(all_geoloc_values))   

    geo_loc_dict = {}
    for i, loc in enumerate(all_geoloc_values):
        if ':' in loc:
            query = loc.split(':')[0]
        else:
            query = loc
        print(f'Processing {i+1}/{len(all_geoloc_values)}')

        if 'not applicable' in query.lower() or \
            'unknown' in query.lower():
            country = None
        else:
            country = rate_limited_get_country(query)

        if country:
            print(f'\t{loc} => {country}')
        else:
            print(f'\tFailed to get country for "{loc}"')

        geo_loc_dict[loc] = country

    with open('./geo_loc_dict.json', 'w') as f:
        f.write(json.dumps(geo_loc_dict, indent=4))
else:
    geo_loc_dict = json.load(open('./geo_loc_dict.json', 'r'))


# Update geoloc values
for pmc_dir in tqdm.tqdm(glob.glob(os.path.join(Config.DATA_DIR, 'PMC*'))):
    pmc_number = pmc_dir.split('/')[-1]
    if not filemanager.check_if_samples_json_exists(pmc_number):
        continue
    samples = filemanager.load_samples_json(pmc_number)

    for s in samples:
        if s.get(target_key) and s.get(target_key).get('Aligned'):
            loc = s[target_key]['Aligned']
            if loc in geo_loc_dict:
                country = geo_loc_dict[loc]
                s[target_key]['Converted'] = loc
                s[target_key]['Aligned'] = country
            else:
                s[target_key]['Converted'] = loc
                s[target_key]['Aligned'] = None
    # save samples
    filemanager.update_samples_json(pmc_number, samples)






