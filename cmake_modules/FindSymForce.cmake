FIND_LIBRARY(SYMFORCE_GEN_LIBRARY symforce_gen PATHS "/usr/local/lib")
FIND_LIBRARY(SYMFORCE_OPT_LIBRARY symforce_opt PATHS "/usr/local/lib")
FIND_LIBRARY(SYMFORCE_SLAM_LIBRARY symforce_slam PATHS "/usr/local/lib")
set(symforce_INCLUDE_DIRS "/usr/local/include/symforce")
set(symforce_LIBRARIES ${SYMFORCE_GEN_LIBRARY} ${SYMFORCE_OPT_LIBRARY}
    ${SYMFORCE_SLAM_LIBRARY})
set
