# Sprint M04 Observations

Name: Abdullah Mamun

I used the `data/universities.csv` dataset and built two semantic axes.  
Axis 1 is `Access/teaching` (negative pole) versus `Prestige/selective` (positive pole).  
Axis 2 is `Arts/humanities` (negative pole) versus `Technical/STEM` (positive pole).  
I picked these because they capture two different aspects of universities: institutional role and academic emphasis. The axis diagnostics printed by the script show good separation between each pole pair, and the two axes are not strongly aligned.

In the scatter, points on the right side tend to be institutions discussed in more elite/selective language, while points on the left side lean toward accessible or teaching-focused language. The vertical direction separates schools with technical framing from schools described with liberal-arts framing. The visual uses color for `region` and shape for `type`, so both categorical attributes are visible at once. I used a colorblind-safe palette and avoided red-green pairing.

The most surprising pattern is that some schools expected to be very technical are not always at the highest technical score, while a few less famous schools still appear strongly technical based on name semantics. This shows that embedding-based maps capture language signals, not ranking tables. In other words, the map reflects how names and phrases relate in embedding space, which can differ from our ranking-based intuition.

A good third axis would be `Research intensity` versus `Teaching focus` using poles such as `doctoral research`, `lab funding`, `publication output` versus `undergraduate teaching`, `small class size`, `student mentoring`. That axis would likely separate institutions that now overlap in this 2D projection and add a clearer story about mission differences.
