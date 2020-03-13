import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Weldext')
    parser.add_argument('action', choices=['train', 'test', 'docs'], help='选取一个 Action')
    args = parser.parse_args()


    if args.action == 'train':
        pass
    elif args.action == 'test':
        pass
    elif args.action == 'docs':
        from docs import docs_dev
        docs_dev()
