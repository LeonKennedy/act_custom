import cv2
import zarr


def show(paht: str):
    root = zarr.open(paht, 'r')

    L = root.meta.episode_ends[-1]
    image = root.data.img
    for i in range(L):
        cv2.imshow("TOP", image[i, 0])
        cv2.imshow("LEFT", image[i, 2])
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    show("train.zarr")
