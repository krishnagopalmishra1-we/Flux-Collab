import json
with open('Colab_Launcher.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        for i, line in enumerate(cell['source']):
            if 'time.sleep(5)' in line:
                cell['source'][i] = '''print("\u23f3 Waiting for server to initialize (can take 20-30s)...")
import urllib.request
for _ in range(30):
    if server_process.poll() is not None: break
    try:
        urllib.request.urlopen("http://127.0.0.1:7860")
        break
    except:
        time.sleep(1)
'''
                print('Replaced sleep with smart polling')
                break
with open('Colab_Launcher.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
print('Done')
