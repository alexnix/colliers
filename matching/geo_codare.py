import requests

token = "&key=AIzaSyASwthI1dzFN5q0tD7wqAvAnNus0idvgWU"
url = "https://maps.googleapis.com/maps/api/geocode/json?address="

def geo_codare_db(df):
    for index, row in df.iterrows():
        address = row['Oras'] + '+' + row['Nume Artera'] + '+' + str(row['Numar'])
        get_url = url + address.replace(' ', '+') + token
        r = requests.post(url = get_url)
        data = r.json()
        latitude = data['results'][0]['geometry']['location']['lat']
        longitude = data['results'][0]['geometry']['location']['lng']
        df.at[index, 'Latitudine'] = latitude
        df.at[index, 'Longitudine'] = longitude
