import argparse
import os

def link(dst, src):
    dst = os.path.join(os.getcwd(), os.path.normpath(dst))
    src = os.path.join(os.getcwd(), os.path.normpath(src))
    if not os.path.exists(src):
        print(src + " doesn't exist... skipping")
        return
    if os.path.exists(dst):
        os.remove(dst)
    os.symlink(src, dst)

def main():
    """ main """
    parser = argparse.ArgumentParser(
                    prog='symlink',
                    description='symlink multiple path')
    parser.add_argument('--file', dest='filepath')

    args = parser.parse_args()

    with open(args.filepath, 'r') as file:
        for line in file.readlines():
            line = line.strip()
            (dst, src) = line.split(' ')
            link(dst, src)

if __name__=='__main__':
    main()
