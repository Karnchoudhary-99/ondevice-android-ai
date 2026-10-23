plugins {
    kotlin("jvm") version "2.0.21"
    application
}

kotlin {
    jvmToolchain(17)
}

application {
    mainClass.set("ai.ondevice.demo.MainKt")
}

dependencies {
    implementation(project(":core"))
}
