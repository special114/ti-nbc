import matplotlib.pyplot as plt

def visualize(X, Y, y_pred, args):
    if not args.visualize:
        return
    if X.shape[1] == 2:
        plt.subplot(1, 2, 1)
        plt.scatter(X[:, 0], X[:, 1], c=y_pred)
        plt.title('Predicted clusters')

        plt.subplot(1, 2, 2)
        plt.scatter(X[:, 0], X[:, 1], c=Y)
        plt.title('Original clusters')

        if args.save_plot:
            file_name = 'plot.png'
            plt.savefig(file_name)
            print(f'Figure successfully saved in file {file_name}')
        else:
            plt.show()
    # elif X.ndim == 3:
    #     plt.scatter(X[:, 0], X[:, 1], X[:, 2])
    #     plt.show()
