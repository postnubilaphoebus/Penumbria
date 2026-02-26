# Codifying the train and val indices (stardist):

# Mouse skull
# dedicated test image
# Fold 0: seed = 0, train_index = 0, val_index = 1
# Fold 1: seed = 1, train_index = 1, val_index = 0
# Fold 2: seed = 2, train_index = 0, val_index = 1

# Paryhale
# test images sampled
# Fold 0: seed = 0, train_index = 3, 4, 5, val_index = 2 test_index = 0, 1
# Fold 1: seed = 1, train_index = 0, 4, 5, val_index = 1 test_index = 2, 3
# Fold 2: seed = 2, train_index = 1, 2, 3, val_index = 0 test_index = 4, 5

# Platynereis-Nuclei-CBG
# test images sampled
# Fold 0: seed = 0, train_index = 3, 4, 5, 6, val_index = 7, 8 test_index = 0, 1, 2
# Fold 1: seed = 1, train_index = 0, 1, 2, 8, val_index = 6, 7 test_index = 3, 4, 5
# Fold 2: seed = 2, train_index = 0, 1, 3, 4, val_index = 2, 5 test_index = 6, 7, 8

# Arabidopsis (CANCELED DUE TO PROCESSING TIME!)
# dedicated test image
# Fold 0: seed = 0, train_index = 0, 1, 2, 3, 4, 5, 6, 7, 8 val_index = 9 #9, 10
# Fold 1: seed = 1, train_index = 2, 3, 4, 5, 6, 7, 8, 9, 10 val_index =  0 #0, 1
# Fold 2: seed = 2, train_index = 0, 1, 2, 3, 6, 7, 8, 9, 10 val_index = 4 #4, 5

# mouse_organoid_cbg
# sampled test image
# Fold 0: seed = 0, train_index = [ 11,  82,  79,  33,  96,   2,   3,  14,   6,  72,  21,  94,  88,
#         53,  17,  43,  86,  84,  52,   1,  41,  40,  76,  87,  34,  99,
#         30,  15,  60,  81,  78,  56,  70, 101,  24,  50,   0,  69,  22,
#         80,  75,  28,  93,  12,  97,  38,  85,  57,   7,  51,  63, 100,
#         74,  67, 107,  95,  31,   5,  66,  19,  77,  48,  39,  13,  16,
#         49,  58,  37,  10,  44,  59,  65,  83,  47,  18,  36,  29,  68,
#        106, 105,  90,  71,  46] ,
# # val_index = [32, 62, 89, 4, 42, 23, 98, 54,  8, 45, 25, 35, 55, 20, 91],
# test = [ 73,  64, 102, 103,  61, 104,   9,  27,  92,  26]
# Fold 1: seed = 1, train_index = [ 92,  57,  56,  62,  17,  11,  99,  36,  25, 105,  97, 106,  22,
        # 10, 103,  58,  31,  50,  74,  85,  39,  38,  55,   3,  86,  79,
        # 52,  81,  30,  14,  98,  40,  35,  82,  65,  87,  54,  88,  20,
        # 91,  43,  67,  12,  77, 100,  29,  72,  28,  46,  15,  66,  95,
        # 19,   2,  83,  13,  53,   0,  78,  94,   6,  48,  69,   4,  64,
        # 24,  75,  21,  47,   7,  89,  42,  73, 104,  37,  33,  45,  16,
        # 34,  18,  23,  76,  32]
# val_index = [ 41, 101,   9,  68,   1, 107,  26, 102,  80,  96,  51,  93,  71, 61,  27]
# test = [84,  8,  5, 49, 90, 44, 63, 70, 59, 60]
# Fold 2: seed = 2, 
# train_index = [ 86,  12,  16,  98,   7,  15, 107,  22,  60,  36,  42,  58,  80,
        # 88,  77,  59,  62,  19,  67,  84, 101, 104,  89,  90,  38,  24,
        #  8,   9,  35,  39,  44,  37, 103,  83,  18,   3,  13,  28,  23,
        # 29,  99,  20,  51,  72,  85,  34,  93,  21,  53,  54,  47,   5,
        # 45,  74,  75,  33,  10, 102,  57,  76,  66,  55,  49,  43,  52,
        #  1,  71,  87,  25,  14,  11,  82,  61,  17, 100,  30,  50,  81,
        # 95,  96,  78,  73,  26]
# val_index = [ 91,  65,  92,  48, 106,  64,  46,  56,  68,  97,   4,   6,  31, 63,  94]
# test index = [  0,  70,  69,  40,  79,  27,   2, 105,  41,  32]


# Quote by chatgpt: "That’s extremely well thought out — and what you’re describing is one of the most promising design philosophies in modern segmentation"