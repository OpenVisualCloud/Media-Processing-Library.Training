To create a directory structure recognizable by FAIME, please follow the below steps.

Replace $VARIABLES with values relevant to your use case.

1) Put all video and image files in a target "media directory".
2) Run the following command "python FAIME/project_builder/project_builder.py --media_folder=$MEDIA_FOLDER --project_folder=$PROJECT_FOLDER --img_per_video=$IPV
3) Replace the --project_folder variable in any of the above configs with $PROJECT_FOLDER
4) Run "python FAIME/FAIME.py --cfg=$config"