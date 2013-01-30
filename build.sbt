name := "machine learning"

version := "0.1"

scalaVersion := "2.10.0"

libraryDependencies  ++= Seq(
            "org.scalanlp" %% "breeze-math" % "0.2-SNAPSHOT",
            "org.scalanlp" %% "breeze-learn" % "0.2-SNAPSHOT",
//            "org.scalanlp" %% "breeze-process" % "0.2-SNAPSHOT",
            "org.scalanlp" %% "breeze-viz" % "0.2-SNAPSHOT"
)

resolvers ++= Seq(
            "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/"
)
