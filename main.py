from time import time

sen = "python programming language"

print(">>",sen)

start = time()
givensen =input(">> ")
end = time()

total_characters = len(sen)
total_characters_given = len(givensen)
r = min(total_characters , total_characters_given)
corret_characters = 0

for i in range(r):
    if sen[i] == givensen[i] :
        corret_characters += 1


time_taken = round((end-start) , 2)

cpm = round((corret_characters/time_taken)*60)

wpm = cpm/4

acc = (corret_characters/total_characters)*100

print("CPM:" ,cpm )
print("WPM:" ,wpm)
print("Acc:" ,acc)

print("correct_characters :", corret_characters)
print("Time Taken :" , time_taken)