import io

import requests
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from httplib2 import Http
from oauth2client import file, client, tools

try :
    import argparse
    flags = argparse.ArgumentParser(parents=[tools.argparser]).parse_args()
except ImportError:
    flags = None

SCOPES = 'https://www.googleapis.com/auth/drive'

store = file.Storage('storage.json')
creds = store.get()

if not creds or creds.invalid:
    print("make new storage data file ")
    flow = client.flow_from_clientsecrets('client_secret_995519708594-f2q6qo1kovqseqtajalpmcd0674r0sa0.apps.googleusercontent.com.json', SCOPES)
    creds = tools.run_flow(flow, store, flags) if flags else tools.run(flow, store)

service = build('drive', 'v3', credentials=creds)

file_id = '1BGtjtS1zvAfXxwuhcrrWyoitWgUBb4de'
request = service.files().get_media(fileId=file_id)
response = requests.get("https://www.googleapis.com/drive/v3/files/1BGtjtS1zvAfXxwuhcrrWyoitWgUBb4de")
print(response)
print(response.content)
print(response.text)
fh = io.BytesIO()
downloader = MediaIoBaseDownload(fh, request)
done = False
while done is False:
    status, done = downloader.next_chunk()
    print(status, done)
    print ("Download %d%%." % int(status.progress() * 100))