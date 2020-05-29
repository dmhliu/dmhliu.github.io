---
layout: post
title: Where Do You Go, Bike Share?
subtitle: Mapping bike share locations and utilization in San Francisco
cover-img: /assets/img/station_background.jpg
tags: [bikeshare, data, transportation, bay area, geopandas, visualization]
---

If you're lucky enough to live in a bike share city, but aren't a bike-sharer, you've probably had and encounter with one. Since the inception of Ford's Gobike sponsorship of Bay Wheels in SF around 2017, the sprawling docking stations periodically buzz with helmeted urbanites launching into the streets and sidewalks, on their way to.. where, exactly?
 
    Although I am a 'bike person--' there may be a fixie hanging from my ceiling-- this past year I have been driving an electric minivan-load of elementary school kids through an infite loop of pickup, drop-off and pickup again. I have more in common with the Muni driver than the average bike sharer. 
     Maybe I just don't align with the bikeshare model. But I've often wondered while dodging or swerving , *'whither goest thou, boosted biker? And whence returnst thou in the eve?'*
The answer to these musings were the lofty goals at the outset of this investigation of the [Bay Wheels data set](https://www.lyft.com/bikes/bay-wheels/system-data).
   

## Going nowhere with zero distance trips: 
![zero dist trips ](/assets/img/eda_zero_dist_trips.jpg)
The [dataset](https://www.lyft.com/bikes/bay-wheels/system-data) presents a few problems with outlier data. After plotting a few stats, its easy to see there are a *lot* of long trips, and also many that start and end at the same station. Some of the latter are surely the result of tourists or other non-commuters. When we look at the zero distance trips by station, we can see that there are definite patterns. Some stations have a much higher incidence. These stations appear to be busy downtown locations, perfect for the tourist use case. The problem remains that these types of trips skew the data while giving us litle insight other than their number and frequency. And they certainly don't answer the where are you going question, so I'm going to summarily ignore them from here on. In addition, after noting that data for San Jose, Oakland, and Berkely is present we are going to focus on trips that start and end in the city of San Francisco.

## Extreme Biking? The overnight trip problem:


![test image from repo](/assets/img/duration_stats.jpg)

It's not immediately clear why there are some many extremely long trips. Are the bikes losing connectivity before the end of the trips, or its an artifact of the process where wayward machines are spirited away in a van and returned home? Given the theoretical [costs](https://www.lyft.com/bikes/bay-wheels/pricing) associated with these trips, it seems unlikely that they represent actual trip durations in many cases.




 The preceding plot certainly presents an interesting pattern: what looks like a fairly normal distribution centered around midday is overshadowed by a complementary cluster of long trips separated by a very empty space.
Looking at the distance in hours, it becomes apparent that the 'blank' space would contain trips that ended in the wee hours of the morning. This implies, but does not prove that the superior cluster may be the result of renters who keep their bicycles until the next morning. There may be other explanations for the patterns in trip end times, but we can't decide anything at this point, other than that the period between 3 and 4 am is the most unpopular time to start and end a trip. 

## Where is the Where?

![test image from repo](/assets/img/baywheels2017-29_final.gif)
