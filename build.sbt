name := "fastKDE"

version := "0.1"

scalaVersion := "2.12.4"

classpathTypes += "maven-plugin"

libraryDependencies ++= Seq(
  "org.nd4j" % "nd4j-native-platform" % "0.9.1",
  "org.scalatest" % "scalatest_2.12" % "3.0.5" % "test"
)
