#!/bin/bash

diff golden_stdout.txt bad_output.txt > stdout_diff.log
if [ -s stdout_diff.log ]
then
  ACCEPT=1
  diff --side-by-side --suppress-common-lines -W 1024 <(awk '/solution/{getline;print}' golden_stdout.txt) <(awk '/solution/{getline; print}' bad_output.txt) > solution_diff.log
fi
if [ -s solution_diff.log ]
then
while read line
  do
    #echo "New diff line: $line"
    string=`echo $line | tr "|" " "`
    echo "New line trnsformed: $string"
    IFS=' ' read -ra ITEMS <<< "$string"
    echo "ITEMS: ${ITEMS}"
    num_items_=`echo ${#ITEMS[@]}`
    num_items=`echo $(($num_items_/2))`
    i=0
    while [ $i -lt $num_items ]
    do
      j=`echo $(($i+$num_items))`
      echo "i=$i, j=$j, ITEMS[$i]=${ITEMS[$i]} and ITEMS[$j]=${ITEMS[$j]}"
      if [[ ${ITEMS[$i]} == ${ITEMS[$j]} ]]
      then
        echo "Continuing..."
        i=`echo $(($i+1))`
        continue
      elif [[ ${ITEMS[$i]} == 0.00 ]]
      then
        rate=`echo "scale=10;((${ITEMS[$j]}/0.002)*100)" | bc`
      else
        echo "Computing:.. scale=10;(((${ITEMS[$i]}-(${ITEMS[$j]}))/${ITEMS[$i]})*100)" 
        rate=`echo "scale=10;(((${ITEMS[$i]}-(${ITEMS[$j]}))/${ITEMS[$i]})*100)" | bc`
      fi
      echo "rate is: $rate" 
      value_pos=`echo "$rate < 0" | bc`
      if [ $value_pos = "1" ]
      then
        rate=`echo "-($rate)" | bc`
      fi
      acceptable=`echo "$rate<=2" | bc`

      if [ $acceptable = "0" ]
      then
        echo "NOT ACCEPTABLE"
        ACCEPT=0
        break 2
      fi
      i=`echo $(($i+1))`
    done
  done < solution_diff.log
  if [ $ACCEPT = 1 ]; then
    mv solution_diff.log special_check.log
  else
    rm solution_diff.log
  fi
fi


