use keyring_core::Entry;
use secrecy::{SecretBox, zeroize::Zeroize};
use serde::{Deserialize, Serialize, Serializer};
use std::sync::LazyLock;

pub use secrecy::{ExposeSecret, SerializableSecret};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(transparent)]
pub struct SecretString(SecretBox<SecretValue>);

impl Default for SecretString {
    fn default() -> Self {
        Self::from(String::new())
    }
}

impl From<String> for SecretString {
    fn from(value: String) -> Self {
        Self(SecretBox::new(Box::new(SecretValue(value))))
    }
}

impl From<&str> for SecretString {
    fn from(value: &str) -> Self {
        Self::from(value.to_owned())
    }
}

impl ExposeSecret<str> for SecretString {
    fn expose_secret(&self) -> &str {
        &self.0.expose_secret().0
    }
}

#[derive(Clone, Deserialize)]
#[serde(transparent)]
struct SecretValue(String);

impl Zeroize for SecretValue {
    fn zeroize(&mut self) {
        self.0.zeroize();
    }
}

impl secrecy::CloneableSecret for SecretValue {}

impl Serialize for SecretValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str("[REDACTED]")
    }
}

impl SerializableSecret for SecretValue {}

const SERVICE: &str = "koharu";

static CREDENTIAL_STORE: LazyLock<Result<(), String>> = LazyLock::new(|| {
    #[cfg(target_os = "linux")]
    {
        linux_keyutils_keyring_store::Store::new()
            .map(|store| keyring_core::set_default_store(store))
            .map_err(|error| error.to_string())
    }

    #[cfg(not(target_os = "linux"))]
    Ok(())
});

/// Load a Koharu secret by key.
///
/// In container/headless environments the platform credential store may not
/// be available (the Linux kernel keyring needs system calls that container
/// runtimes usually block, and it forgets everything when the container
/// restarts). Treat any credential-store read failure as "no credential" and
/// look at the environment instead: the secret of provider `openai-compatible`
/// is the variable `KOHARU_OPENAI_COMPATIBLE_API_KEY`, which a hosted space can
/// set as a secret. This allows deployments without a platform keyring to use
/// remote providers, and deployments that only use local translation models to
/// run without either.
pub fn get(key: &str) -> anyhow::Result<Option<SecretString>> {
    let stored = entry(key)
        .ok()
        .and_then(|entry| entry.get_password().ok())
        .filter(|value| !value.trim().is_empty())
        .map(SecretString::from);
    Ok(stored.or_else(|| from_environment(key)))
}

/// The name of the environment variable that holds the secret of `key`.
fn environment_name(key: &str) -> String {
    let key: String = key
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_uppercase() } else { '_' })
        .collect();
    format!("KOHARU_{key}_API_KEY")
}

fn from_environment(key: &str) -> Option<SecretString> {
    let value = std::env::var(environment_name(key)).ok()?;
    let value = value.trim();
    (!value.is_empty()).then(|| SecretString::from(value))
}

/// Store a Koharu secret by key.
pub fn set(key: &str, secret: &SecretString) -> anyhow::Result<()> {
    use anyhow::Context;
    entry(key)
        .and_then(|entry| entry.set_password(secret.expose_secret()).map_err(Into::into))
        .with_context(|| {
            format!(
                "could not store the credential in the platform credential store; set the \
                 environment variable {} instead",
                environment_name(key)
            )
        })
}

/// Delete a Koharu secret by key. Missing credentials are treated as success.
pub fn delete(key: &str) -> anyhow::Result<()> {
    match entry(key)?.delete_credential() {
        Ok(()) | Err(keyring_core::Error::NoEntry) => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn entry(key: &str) -> anyhow::Result<Entry> {
    match &*CREDENTIAL_STORE {
        Ok(()) => {}
        Err(error) => anyhow::bail!("failed to initialize Linux Keyutils: {error}"),
    }

    #[cfg(target_os = "linux")]
    let entry = Entry::new(SERVICE, key)?;

    #[cfg(not(target_os = "linux"))]
    let entry = keyring::Entry::new(SERVICE, key)?.inner;

    Ok(entry)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_environment_variable_is_named_after_the_provider() {
        assert_eq!(
            environment_name("openai-compatible"),
            "KOHARU_OPENAI_COMPATIBLE_API_KEY"
        );
        assert_eq!(environment_name("openai"), "KOHARU_OPENAI_API_KEY");
    }

    #[test]
    fn a_secret_comes_from_the_environment_when_the_store_has_none() {
        // SAFETY: the only test that touches this variable.
        unsafe { std::env::set_var("KOHARU_TEST_PROVIDER_API_KEY", "  nvapi-123 \n") };
        let secret = get("test-provider").unwrap().unwrap();
        assert_eq!(secret.expose_secret(), "nvapi-123");
        unsafe { std::env::remove_var("KOHARU_TEST_PROVIDER_API_KEY") };
        assert!(from_environment("test-provider").is_none());
    }
}
