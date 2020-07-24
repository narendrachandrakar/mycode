import hvac
import csv
import re

client = None
all_list = []
vault_url = 'http://13.56.19.116:8200'
token = 's.waRe4tjVzHIYxzJBJXt6kYsC'


client = hvac.Client(url = vault_url, token = token, verify=False)

# vault_client = hvac.Client(
#                 url=vault_url,
#                 cert=certs,
#         )
res = client.is_authenticated()
print("res:", res)
if res != True:
    print('Error! Please check URL & User name and password')
    exit(1)


with open('AddUpdatePath', 'r') as fp:
    lines = fp.readlines()
    for line in lines:
        out = {}
        for tt in line.split(","):
              key = tt.split("=")[0]
              key = key.split()[0]
              value = tt.split("=")[1]
              value = value.split()[0]
              #print(key)
              #print(value)
              if key == 'path':
                  fullpath = value
              else:
                out[key]=value.replace("\n", '')
        print(out)
       # create_response = client.secrets.kv.v2.create_or_update_secret(path=fullpath, secret=out,)


# read_secret_result = client.secrets.kv.v1.read_secret(
#     path=secret_path,
#     mount_point=mount_point,
# )
# print(read_secret_result)

#client.write("hello", hello='ttt')


def read_all(key_path):
    list_response = client.secrets.kv.v2.list_secrets(
        path=key_path
    )
    return  list_response

def get_paths():
    list_folders = read_all('')
    tmp = list_folders['data']['keys']
    #print(tmp)
    for vault in tmp:
        traverse(vault)

def traverse(path):
    regex = re.compile('[@_!#$%^&*()<>?/\|}{~:]')
    if (regex.search(path) == None):
        #print(path)
        all_list.append(path)
        return

    result = read_all(path)

    for secret in result['data']['keys']:
        #print(secret)
        regex = re.compile('[@_!#$%^&*()<>?/\|}{~:]')
        if (regex.search(secret) != None):
            tmp1 = path + secret
            traverse(tmp1)
        else:
            full_path = None
            full_path = path + secret
            all_list.append(full_path)
            #print(full_path)
    #print("++++++ Final list++++++++++")
    #print(all_list)


def get_key_value_details():
    with open("output.csv", 'w') as csvFile:
        fieldnames = ['path', 'key', 'value']
        writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
        writer.writeheader()
        for full_path in all_list:
            read_response = client.secrets.kv.read_secret_version(path=full_path)
            for key, value in read_response['data']['data'].items():
                writer.writerow({'path': full_path, 'key': key, 'value': value})

def display_output():
    with open("output.csv", 'r') as csvFile:
        csv_reader = csv.DictReader(csvFile)
        for row in csv_reader:
            #print(row)
            if row["key"] == 'password':
                row["value"] = "*********"
            print( f'\tpath: {row["path"]} , key: {row["key"]}, value: {row["value"]}')




if __name__ == "__main__":
    get_paths()
    get_key_value_details()
    display_output()