"""
read markdown file from path stored in env variable CS486_PATH
then, write a new markdown file with the same contents, but with one modification:
convert all links that look like
![400](assets/lec18.2.png)
to
<img src="assets/lec18.2.png" width="400">

also skip the first line of the file

Copy any files from the assets directory in the extracted directory path to the current working directory's assets folder if they do not already exist.
"""

import os
import re
import shutil

def convert_md():
    path = os.environ['CS486_PATH']
    directory_path = os.path.dirname(path)
    assets_src_path = os.path.join(directory_path, 'assets')
    assets_dest_path = os.path.join(os.getcwd(), 'assets')

     # Ensure the destination assets directory exists
    os.makedirs(assets_dest_path, exist_ok=True)

    # Copy files from source assets directory to destination assets directory
    if os.path.exists(assets_src_path):
        for filename in os.listdir(assets_src_path):
            src_file = os.path.join(assets_src_path, filename)
            dest_file = os.path.join(assets_dest_path, filename)
            if not os.path.exists(dest_file):
                shutil.copy(src_file, dest_file)

    with open(path, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for i, line in enumerate(lines):
        if i == 0: continue
        new_line = re.sub(r'!\[(\d+)\]\((.+)\)', r'<img src="\2" width="\1">', line)
        new_lines.append(new_line)
    with open('README.md', 'w') as f:
        f.writelines(new_lines)

if __name__ == '__main__':
    convert_md()
