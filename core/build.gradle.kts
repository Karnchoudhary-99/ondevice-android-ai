plugins {
    kotlin("jvm") version "2.0.21"
}

kotlin {
    jvmToolchain(17)
}

dependencies {
    testImplementation("org.junit.jupiter:junit-jupiter:5.10.3")
}

tasks.test {
    useJUnitPlatform()
}
