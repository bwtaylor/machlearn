package com.bwtaylor

import breeze.linalg.DenseVector
import util.Random
import collection.mutable.ListBuffer

object coins {

  val answers = "1=>[d]b, 2=>[d], 3=>[e], 4=>[b], 5=>[c], 6=>[d]c, 7=>[a], 8=>[d], 9=>[a], 10=>[a]b"

  def flip(numCoins: Int, numTrials: Int): DenseVector[Int] = {
    val rand = new Random()
    val coins = DenseVector.zeros[Int](numCoins)
    for (i <- 0 until numCoins)
      coins(i) = (1 to numTrials).map{x => rand.nextInt(2)}.sum
    coins
  }

  def freqs(numCoins: Int, numTrials: Int): (Int, Int, Int) = {
    val coins: DenseVector[Int] = flip(numCoins,numTrials)
    val (c1, cRand, cMin) = (1, new Random().nextInt(numCoins), coins.toArray.indexOf(coins.min) )
    return (coins(c1), coins(cRand), coins(cMin))
  }


  def main(args: Array[String]) = {
    val (v1List, vRandList, vMinList) = (new ListBuffer[Int], new ListBuffer[Int], new ListBuffer[Int])
    val N = 100000
    for ( i <- 1 to N ) {
      val (v1, vRand, vMin) = freqs(1000,10)
      v1List += v1
      vRandList += vRand
      vMinList += vMin
    }

    println("v1Avg=%f vRandAvg=%f vMinAvg=%f".format(v1List.sum/(N*1.0), vRandList.sum/(N*1.0), vMinList.sum/(N*1.0)))
  }

}
