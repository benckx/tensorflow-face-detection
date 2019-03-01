# About
Detect faces in folders of images and write the resulting information into a CSV.

See fork parent for more info.

# Usage

Simply pass the folders containing the images:

    $ python inference_image_face.py <image_folder1> <image_folder2> ...

By default, it will pick up the test_images folder.

    $ python inference_image_face.py

Which will result in the following:

| Image Location  | Score| X1  | Y1| X2  | Y2 |
| --------------------------------------------------------- | --- | --- | --- | --- | --- |
|test_images/nabokov.jpg                                    |0.999|  130|  387|  318|  387|
|test_images/1280px-Reagan_signs_Martin_Luther_King_bill.jpg|0.982|  819|  292|  892|  292|
|test_images/1280px-Reagan_signs_Martin_Luther_King_bill.jpg|0.980|  630|  263|  700|  263|
|test_images/1280px-Reagan_signs_Martin_Luther_King_bill.jpg|0.940|  305|  268|  388|  268|
|test_images/1280px-Reagan_signs_Martin_Luther_King_bill.jpg|0.926|  960|  315| 1036|  315|
|test_images/1280px-Reagan_signs_Martin_Luther_King_bill.jpg|0.907|  425|  278|  501|  278|
|test_images/1280px-Reagan_signs_Martin_Luther_King_bill.jpg|0.805|  776|  526|  877|  526|
|test_images/subfolder/musk_obama.jpeg                      |0.989|  164|  413|  232|  413|
|test_images/subfolder/musk_obama.jpeg                      |0.970| 1678|  503| 1736|  503|
|test_images/subfolder/musk_obama.jpeg                      |0.968|  465|  495|  517|  495|
|test_images/subfolder/musk_obama.jpeg                      |0.893|  643|  350|  718|  350|
|test_images/subfolder/musk_obama.jpeg                      |0.872|  963|  378| 1033|  378|
