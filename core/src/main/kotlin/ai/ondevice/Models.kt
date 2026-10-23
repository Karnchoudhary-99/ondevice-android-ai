package ai.ondevice

import java.io.File
import java.security.MessageDigest

/**
 * Model management for the Android app: the store on the phone holds one
 * bundled model plus any models downloaded later, and the app always runs
 * the *active* one. This is the JVM twin of that store — same rules, same
 * files — so the download-once / version / integrity logic is tested here
 * and the Android UI only has to call it.
 *
 * Layout inside the store directory:
 * ```
 * <name>.bin    the ODAI model bytes
 * <name>.meta   name, version, sha256, byte size, labels (line format)
 * active.txt    name of the active model (absent = none active)
 * ```
 * Rules: installing a byte-identical model is a no-op (download-once);
 * installing an older-or-equal version over a newer one is rejected
 * (downgrade protection); a source file that fails ODAI validation is
 * rejected before anything is written, so a bad download can never
 * clobber a good installed model.
 */
data class InstalledModel(
    val name: String,
    val version: Int,
    val sha256: String,
    val bytes: Long,
    val labels: List<String>,
)

/** Result of an install: either a fresh copy, or a no-op for identical bytes. */
sealed interface InstallResult {
    data class Installed(val model: InstalledModel) : InstallResult
    data class AlreadyPresent(val model: InstalledModel) : InstallResult
}

fun sha256Hex(file: File): String {
    val digest = MessageDigest.getInstance("SHA-256")
    file.inputStream().use { stream ->
        val buffer = ByteArray(8192)
        while (true) {
            val read = stream.read(buffer)
            if (read < 0) break
            digest.update(buffer, 0, read)
        }
    }
    return digest.digest().joinToString("") { "%02x".format(it) }
}

class ModelStore(private val dir: File) {

    init {
        dir.mkdirs()
    }

    private fun modelFile(name: String) = File(dir, "$name.bin")
    private fun metaFile(name: String) = File(dir, "$name.meta")
    private fun activeFile() = File(dir, "active.txt")

    /**
     * Install [source] under [name] at [version]. Returns [InstallResult]
     * so callers can tell a fresh install from a download-once no-op.
     */
    fun install(source: File, name: String, version: Int): InstallResult {
        require(name.isNotBlank() && !name.contains(File.separator) && !name.contains("..")) {
            "invalid model name: '$name'"
        }
        require(version > 0) { "version must be positive, got $version" }
        require(source.isFile) { "model source not found: ${source.path}" }
        // Validate before touching the store: a corrupt download must not
        // replace a working model.
        val loaded = ModelFile.load(source)
        val sha = sha256Hex(source)
        val existing = installed(name)
        if (existing != null) {
            if (existing.sha256 == sha) return InstallResult.AlreadyPresent(existing)
            require(version > existing.version) {
                "refusing downgrade of '$name': installed v${existing.version}, offered v$version"
            }
        }
        val model = InstalledModel(name, version, sha, source.length(), loaded.labels)
        // Write via a temp file, then rename: a crash mid-copy never leaves
        // a half-written model in place.
        val temp = File(dir, "$name.bin.tmp")
        source.copyTo(temp, overwrite = true)
        val target = modelFile(name)
        if (target.exists()) target.delete()
        require(temp.renameTo(target)) { "could not install model '$name'" }
        metaFile(name).writeText(
            listOf(
                "name=$name",
                "version=$version",
                "sha256=$sha",
                "bytes=${source.length()}",
                "labels=${loaded.labels.joinToString(",")}",
            ).joinToString("\n") + "\n",
        )
        return InstallResult.Installed(model)
    }

    /** The installed model with [name], or null. Reads the .meta sidecar. */
    fun installed(name: String): InstalledModel? {
        val meta = metaFile(name)
        val bin = modelFile(name)
        if (!meta.isFile || !bin.isFile) return null
        val fields = meta.readLines().associate { line ->
            val split = line.indexOf('=')
            if (split < 0) "" to "" else line.substring(0, split) to line.substring(split + 1)
        }
        val version = fields["version"]?.toIntOrNull() ?: return null
        val sha = fields["sha256"] ?: return null
        val bytes = fields["bytes"]?.toLongOrNull() ?: return null
        val labels = fields["labels"]?.split(",")?.filter { it.isNotEmpty() } ?: return null
        return InstalledModel(name, version, sha, bytes, labels)
    }

    /** Every installed model, sorted by name. */
    fun list(): List<InstalledModel> =
        dir.listFiles { file -> file.name.endsWith(".meta") }
            ?.mapNotNull { installed(it.name.removeSuffix(".meta")) }
            ?.sortedBy { it.name }
            ?: emptyList()

    /** Make [name] the model the app runs. The model must be installed. */
    fun activate(name: String) {
        require(installed(name) != null) { "cannot activate '$name': not installed" }
        activeFile().writeText(name)
    }

    /** The active model, or null when none has been activated yet. */
    fun active(): InstalledModel? {
        val name = activeFile().takeIf { it.isFile }?.readText()?.trim().orEmpty()
        return if (name.isEmpty()) null else installed(name)
    }

    /** The ODAI file of the active model, or null when none is active. */
    fun activeModelFile(): File? = active()?.let { modelFile(it.name) }

    /** Remove [name] from the store; clears the active pointer if it pointed here. */
    fun remove(name: String) {
        require(installed(name) != null) { "cannot remove '$name': not installed" }
        modelFile(name).delete()
        metaFile(name).delete()
        if (activeFile().takeIf { it.isFile }?.readText()?.trim() == name) activeFile().delete()
    }

    /**
     * Integrity check: the stored bytes still hash to the manifest value
     * and still parse as an ODAI model. Returns false (never throws) for
     * tampered or truncated files — the app calls this on startup and
     * falls back to the bundled model when it fails.
     */
    fun verify(name: String): Boolean {
        val model = installed(name) ?: return false
        val file = modelFile(name)
        return try {
            sha256Hex(file) == model.sha256 && ModelFile.load(file).labels == model.labels
        } catch (_: Exception) {
            false
        }
    }
}
