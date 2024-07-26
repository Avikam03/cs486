"""
read markdown file from path stored in env variable CS486_PATH
then, write a new markdown file with the same contents, but with one modification:
convert all links that look like
![400](assets/lec18.2.png)
to
<img src="assets/lec18.2.png" width="400">

also skip the first line of the file

remove the current working directory's assets folder if it exists.
copy the entire assets directory from the extracted directory path to the current working directory.
"""

import os
import re
import shutil

def convert_md():
    path = os.environ['CS486_PATH']
    directory_path = os.path.dirname(path)
    assets_src_path = os.path.join(directory_path, 'assets')
    assets_dest_path = os.path.join(os.getcwd(), 'assets')


    # Remove the destination assets directory if it exists
    if os.path.exists(assets_dest_path):
        shutil.rmtree(assets_dest_path)

    # Copy the entire assets directory from source to destination
    if os.path.exists(assets_src_path):
        shutil.copytree(assets_src_path, assets_dest_path)
     
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
